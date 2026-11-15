//! SECAM method 1 colour-under restoration. The recorder took the bell
//! pre-emphasis off the studio chroma block, passed the block through a
//! divide-by-4 counter and put a bell pre-emphasis of its own on the result,
//! so the tape holds rest carriers at foB/4 and foR/4 with the FM deviation
//! divided by four as well, under an anti-bell centred between them.
//! Playback mirrors the recorder stage by stage: the bell that undoes the
//! tape's anti-bell, a limiter, the carrier phase multiplied back up, and the
//! BT.470 anti-bell on the restored block. Carrier and deviation scale
//! together, so tape timebase error self-corrects and there is no conversion
//! LO to servo.
//!
//! The line blanking is then rewritten with the rest carrier, because what
//! the tape holds there is the divider settling after the blanking edge
//! rather than the rest carrier the decoders downstream zero their
//! discriminators on.

use std::f64::consts::{PI, TAU};

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use super::*;
use crate::spec::{next_fast_len, sosfiltfft};

/// Centre of the HF pre-emphasis of the chroma block (ITU-R BT.470); the
/// tape's own pre-emphasis sits at the same place counted down.
const BELL_F0_HZ: f64 = 4_286_000.0;
/// Carrier envelope, relative to the field median, the limiter starts at.
/// Below it the amplitude passes through, so a carrier that collapses
/// (dropouts, no colour) reaches the downstream colour killers as a collapse
/// rather than as full-scale noise.
const LIMITER_KNEE: f32 = 0.5;
/// Rest carriers of the D'R and D'B lines (ITU-R BT.470).
const REST_HZ: [f64; 2] = [4_406_250.0, 4_250_000.0];

/// The blanking splice opens this long before the active video ends, ahead
/// of what the block band spreads back from the blanking edge.
const SPLICE_LEAD_US: f64 = 0.85;
/// The synthesized carrier fades in over the outgoing one for this long
/// before the splice opens.
const FADE_US: f64 = 0.45;
/// Glide from the outgoing carrier's frequency to its line's rest carrier.
const RAMP_US: f64 = 1.13;
/// Carrier measurement window on either side of the splice.
const MEASURE_US: f64 = 1.8;
/// Step from one line's rest carrier to the next's, from the incoming line's
/// start: over the sync tip.
const STEP_US: [f64; 2] = [0.45, 2.26];
/// Frequency excursion closing the phase between the two lines' carriers,
/// from the incoming line's start. It ends with the sync pulse, so the back
/// porch holds the bare rest carrier.
const BUMP_US: [f64; 2] = [-0.9, 4.96];
const BUMP_TAPER_US: f64 = 0.68;
/// Largest excursion the closure may need, for any phase between the lines.
const BUMP_MAX_HZ: f64 = 350e3;
/// Back porch stretch the line alternation is read from, counted back from
/// the active video start.
const PORCH_US: [f64; 2] = [3.7, 0.3];

/// HF pre-emphasis (anti-bell) network centred on `f0` at `freq_hz`:
/// (1 + j16F) / (1 + j1.26F) with F = f/f0 - f0/f (BT.470-6 table 2 item
/// 2.13). Its inverse is the bell.
fn anti_bell(freq_hz: f64, f0: f64) -> Complex64 {
    let bell_f = freq_hz / f0 - f0 / freq_hz;
    Complex64::new(1.0, 16.0 * bell_f) / Complex64::new(1.0, 1.26 * bell_f)
}

/// The playback chain over a whole field, in the spectrum of the field
/// zero-padded to a fast transform length. Both networks are applied as the
/// complex responses of the analog circuits rather than as zero-phase gains:
/// the bell has to take the phase of the recorder's anti-bell back out along
/// with its amplitude, and the anti-bell has to hand the downstream bell the
/// phase it will take out.
pub(crate) struct Method1Playback {
    r2c: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    forward: Arc<dyn Fft<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    /// Per unique bin: the bell around the under carriers, doubled into the
    /// analytic signal.
    bell: Vec<Complex32>,
    /// Per unique bin: the anti-bell within the block band, scaled to the
    /// output amplitude through the transform pair.
    anti_bell: Vec<Complex32>,
    mult: u32,
    blanking: Blanking,
}

impl Method1Playback {
    /// For fields of up to `len` samples at `samp_rate`, of a carrier counted
    /// down by `mult`. `rest_amplitude` is the peak output amplitude of the
    /// undeviated carrier at the bell centre; `block_band` band-limits the
    /// restored block the way a forward-backward pass of it would. Lines are
    /// `line_len` samples with the picture over `active_video_us`.
    pub(crate) fn new(
        len: usize,
        samp_rate: f64,
        mult: u32,
        rest_amplitude: f64,
        block_band: &[Sos<f64>],
        line_len: usize,
        active_video_us: [f64; 2],
    ) -> Self {
        let n = next_fast_len(len);
        let bin_hz = samp_rate / n as f64;
        let under_f0 = BELL_F0_HZ / f64::from(mult);
        // Neither band reaches DC or the Nyquist bin, which the analytic
        // signal and the real inverse both want kept apart.
        let per_bin = |response: &dyn Fn(usize, f64) -> Complex64| -> Vec<Complex32> {
            (0..=n / 2)
                .map(|bin| {
                    if bin == 0 || bin == n / 2 {
                        return Complex32::new(0.0, 0.0);
                    }
                    let value = response(bin, bin as f64 * bin_hz);
                    Complex32::new(value.re as f32, value.im as f32)
                })
                .collect()
        };
        let bell = per_bin(&|_, freq| 2.0 / anti_bell(freq, under_f0));
        // A unit phasor of the positive half spectrum comes back from the
        // real inverse at 2n.
        let band = sosfiltfft(block_band, n);
        let scale = rest_amplitude / (2 * n) as f64;
        let anti_bell =
            per_bin(&|bin, freq| anti_bell(freq, BELL_F0_HZ) * (band[bin].norm_sqr() * scale));

        let mut real_planner = RealFftPlanner::<f32>::new();
        let mut planner = FftPlanner::<f32>::new();
        Self {
            r2c: real_planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
            forward: planner.plan_fft_forward(n),
            c2r: real_planner.plan_fft_inverse(n),
            bell,
            anti_bell,
            mult,
            blanking: Blanking::new(samp_rate, line_len, active_video_us, rest_amplitude),
        }
    }

    /// The chroma block restored from the colour-under `chroma`.
    fn restore(&self, chroma: &[f32]) -> Vec<f32> {
        let n = self.r2c.len();
        let len = chroma.len();
        assert!(
            len <= n,
            "chroma of {len} samples does not fit the {n}-sample transform sized for the field"
        );
        let zero = Complex32::new(0.0, 0.0);
        let mut padded = vec![0.0f32; n];
        padded[..len].copy_from_slice(chroma);
        let mut half = self.r2c.make_output_vec();
        self.r2c
            .process(&mut padded, &mut half)
            .expect("r2c forward FFT failed");

        // The under carrier as an analytic signal with the tape's anti-bell
        // taken off, which leaves the constant amplitude the divider put out.
        let mut carrier: Vec<Complex32> = half
            .iter()
            .zip(&self.bell)
            .map(|(&bin, &response)| bin * response)
            .collect();
        carrier.resize(n, zero);
        self.inverse.process(&mut carrier);

        // Limiter and multiplier: the unit phasor raised to the multiplier
        // carries the phase multiplied up, with no angle ever taken.
        let envelope: Vec<f32> = carrier[..len].iter().map(|z| z.norm_sqr().sqrt()).collect();
        let knee = median_radix(&envelope) * LIMITER_KNEE;
        for (z, &amplitude) in carrier.iter_mut().zip(&envelope) {
            *z = if amplitude > 0.0 {
                (*z / amplitude).powu(self.mult) * (amplitude / knee).min(1.0)
            } else {
                zero
            };
        }
        carrier[len..].fill(zero);
        let dr_on_even = self.blanking.dr_on_even(&carrier[..len]);

        self.forward.process(&mut carrier);
        for ((bin, &restored), &response) in half.iter_mut().zip(&carrier).zip(&self.anti_bell) {
            *bin = restored * response;
        }
        self.c2r
            .process(&mut half, &mut padded)
            .expect("c2r inverse FFT failed");
        padded.truncate(len);
        if let Some(dr_on_even) = dr_on_even {
            self.blanking.resynthesize(&mut padded, dr_on_even);
        }
        padded
    }
}

/// The line blanking of the restored block, rewritten from one line's
/// picture to the next's: the outgoing carrier glides to its line's rest
/// carrier over the front porch, steps to the incoming line's over the sync
/// tip and holds it through the back porch, in one phase from the carrier
/// measured before the splice to the one measured after it and at the
/// amplitude the anti-bell gives each frequency.
struct Blanking {
    samp_rate: f64,
    line_len: usize,
    /// Splice start within the outgoing line.
    start: usize,
    /// Splice length, to the incoming line's active video.
    span: usize,
    measure: usize,
    /// Back porch stretch within a line.
    porch: Range<usize>,
    /// Weight of the synthesized carrier over the fade ahead of the splice.
    fade: Vec<f64>,
    /// Per written sample, and one past the last: the running sums of the
    /// glide, the step and the closure excursion, each at unit height.
    sums: Vec<[f64; 3]>,
    /// Peak amplitude of the undeviated carrier at the bell centre.
    rest_amplitude: f64,
    /// Peak amplitude under which each rest carrier is one the limiter let
    /// collapse.
    floor: [f64; 2],
}

impl Blanking {
    fn new(
        samp_rate: f64,
        line_len: usize,
        active_video_us: [f64; 2],
        rest_amplitude: f64,
    ) -> Self {
        let px = |us: f64| (us * samp_rate / 1e6).round() as usize;
        let active_start = px(active_video_us[0]);
        let start = px(active_video_us[1] - SPLICE_LEAD_US);
        let (fade, ramp, taper) = (px(FADE_US), px(RAMP_US), px(BUMP_TAPER_US));
        let measure = px(MEASURE_US);
        assert!(
            fade + measure <= start && start < line_len && px(PORCH_US[0]) <= active_start,
            "active_video_us of {active_video_us:?} us leaves no room to resynthesize the line blanking"
        );
        let next_line = line_len - start;
        let span = next_line + active_start;
        let line_us = line_len as f64 * 1e6 / samp_rate;
        let from_next_line = |us: f64| px(line_us + us) - start;
        let step = from_next_line(STEP_US[0])..from_next_line(STEP_US[1]);
        let bump = from_next_line(BUMP_US[0])..from_next_line(BUMP_US[1]);
        assert!(
            ramp <= step.start.min(bump.start)
                && step.end.max(bump.end) <= span
                && 2 * taper < bump.len(),
            "line blanking of {span} samples is too short to resynthesize"
        );

        let raised_cosine = |k: usize, len: usize| 0.5 - 0.5 * (PI * k as f64 / len as f64).cos();
        // Glide, step and excursion at each written sample; the fade runs on
        // the outgoing carrier as measured.
        let shape = |written: usize| -> [f64; 3] {
            let Some(k) = written.checked_sub(fade) else {
                return [1.0, 0.0, 0.0];
            };
            let glide = if k < ramp {
                1.0 - raised_cosine(k, ramp)
            } else {
                0.0
            };
            let stepped = if k < step.start {
                0.0
            } else if k < step.end {
                raised_cosine(k - step.start, step.len())
            } else {
                1.0
            };
            let excursion = if bump.contains(&k) {
                let edge = (k - bump.start).min(bump.end - 1 - k);
                raised_cosine(edge.min(taper), taper)
            } else {
                0.0
            };
            [glide, stepped, excursion]
        };
        let mut sums = vec![[0.0; 3]; fade + span + 1];
        for written in 0..fade + span {
            let (so_far, here) = (sums[written], shape(written));
            sums[written + 1] = std::array::from_fn(|term| so_far[term] + here[term]);
        }
        assert!(
            BUMP_MAX_HZ * sums[fade + span][2] / samp_rate >= 0.5,
            "the sync pulse is too short to close half a turn of carrier phase"
        );
        Self {
            samp_rate,
            line_len,
            start,
            span,
            measure,
            porch: active_start - px(PORCH_US[0])..active_start - px(PORCH_US[1]),
            fade: (0..fade).map(|k| raised_cosine(k, fade)).collect(),
            sums,
            rest_amplitude,
            floor: REST_HZ.map(|rest_hz| {
                f64::from(LIMITER_KNEE) * rest_amplitude * anti_bell(rest_hz, BELL_F0_HZ).norm()
            }),
        }
    }

    /// Whether the even rows of the multiplied `carrier` are the D'R lines,
    /// from the back porch frequency of the even rows against the odd ones;
    /// `None` when the two do not sit a rest carrier pair apart.
    fn dr_on_even(&self, carrier: &[Complex32]) -> Option<bool> {
        let mut rotation = [Complex32::new(0.0, 0.0); 2];
        for (row, line) in carrier.chunks_exact(self.line_len).enumerate() {
            rotation[row % 2] += line[self.porch.clone()]
                .windows(2)
                .map(|pair| pair[1] * pair[0].conj())
                .sum::<Complex32>();
        }
        let hz = |rotation: Complex32| f64::from(rotation.arg()) * self.samp_rate / TAU;
        let split = hz(rotation[0]) - hz(rotation[1]);
        (split.abs() > (REST_HZ[0] - REST_HZ[1]) / 2.0).then_some(split > 0.0)
    }

    /// Peak amplitude of the carrier in `window`, its frequency, and its
    /// phase at the window centre. The window is correlated against
    /// `rest_hz`: the turn between its two halves is the offset from it.
    fn measure(&self, window: &[f32], rest_hz: f64) -> (f64, f64, f64) {
        let rotor_step = TAU * rest_hz / self.samp_rate;
        let half = window.len() / 2;
        let mut power = 0.0;
        let mut halves = [Complex64::new(0.0, 0.0); 2];
        for (k, &sample) in window.iter().enumerate() {
            let sample = f64::from(sample);
            power += sample * sample;
            halves[usize::from(k >= half)] +=
                Complex64::from_polar(sample, -rotor_step * k as f64);
        }
        let offset_step = (halves[1] * halves[0].conj()).arg() / half as f64;
        let centre = (window.len() - 1) as f64 / 2.0;
        (
            (2.0 * power / window.len() as f64).sqrt(),
            rest_hz + offset_step * self.samp_rate / TAU,
            (halves[0] + halves[1]).arg() + rotor_step * centre,
        )
    }

    /// Rewrite the blanking between the rows of `block`. A splice with no
    /// carrier on either side is left alone.
    fn resynthesize(&self, block: &mut [f32], dr_on_even: bool) {
        let hz_step = TAU / self.samp_rate;
        let (fade, measure) = (self.fade.len(), self.measure);
        let centre = (measure - 1) as f64 / 2.0;
        let written = fade + self.span;
        for row in 0..block.len() / self.line_len {
            let start = row * self.line_len + self.start;
            let end = start + self.span;
            if end + measure > block.len() {
                break;
            }
            let outgoing = usize::from((row % 2 == 0) != dr_on_even);
            let rest_hz = [REST_HZ[outgoing], REST_HZ[1 - outgoing]];
            let (amplitude_out, hz_out, phase_out) =
                self.measure(&block[start - measure..start], rest_hz[0]);
            let (amplitude_in, hz_in, phase_in) =
                self.measure(&block[end..end + measure], rest_hz[1]);
            if amplitude_out < self.floor[outgoing] || amplitude_in < self.floor[1 - outgoing] {
                continue;
            }

            // The phase run from the outgoing carrier, then the excursion
            // that lands it on the incoming one.
            let phase_start =
                phase_out + hz_step * hz_out * (measure as f64 - fade as f64 - centre);
            let (glide, step) = (hz_out - rest_hz[0], rest_hz[1] - rest_hz[0]);
            let open = |k: usize| {
                let [glided, stepped, _] = self.sums[k];
                phase_start + hz_step * (rest_hz[0] * k as f64 + glide * glided + step * stepped)
            };
            let miss = phase_in - hz_step * hz_in * centre - open(written);
            let excursion = ((miss + PI).rem_euclid(TAU) - PI) / self.sums[written][2];

            let phase_at = |k: usize| open(k) + excursion * self.sums[k][2];
            for (k, sample) in block[start - fade..end].iter_mut().enumerate() {
                let phase = phase_at(k);
                let hz = (phase_at(k + 1) - phase) / hz_step;
                let amplitude = self.rest_amplitude * anti_bell(hz, BELL_F0_HZ).norm();
                let carrier = amplitude * phase.cos();
                let weight = self.fade.get(k).copied().unwrap_or(1.0);
                *sample = (f64::from(*sample) * (1.0 - weight) + carrier * weight) as f32;
            }
        }
    }
}

/// The restored chroma block of a field, zeroed up to the line colour comes
/// back on after a field without it. No per-line AGC: the limiter sets the
/// carrier amplitude and the anti-bell shapes it, and normalizing every line
/// to its porch level would flatten the intended foR/foB rest amplitude
/// difference.
pub(super) fn process_chroma(
    field: &DecodedField,
    playback: &Method1Playback,
    chroma: &[f32],
) -> Vec<f32> {
    let mut block = playback.restore(chroma);
    if let Ok(line @ 1..) = usize::try_from(field.burst_start_line()) {
        let blanked = (field.output_row(line) * field.outlinelen).min(block.len());
        block[..blanked].fill(0.0);
    }
    block
}
