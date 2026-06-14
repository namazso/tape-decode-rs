use super::*;
use crate::request::SecamMode;

fn adjust_phase(
    input_data: &[Complex32],
    output_data: &mut [f32],
    input_phase: f64,
    target_phase: f64,
) {
    assert_eq!(input_data.len(), output_data.len());

    let phase_adjustment = target_phase.to_radians() - input_phase.to_radians();
    // The rotation factors are computed once in f64 for accuracy, then the
    // per-sample mix runs in f32 over the already-f32 analytic signal.
    let rotation_re = phase_adjustment.cos() as f32;
    let rotation_im = phase_adjustment.sin() as f32;

    for (input, output) in input_data.iter().zip(output_data.iter_mut()) {
        *output = input.re * rotation_re - input.im * rotation_im;
    }
}

fn acc(
    chroma: &[f32],
    burst_abs_ref: f32,
    burststart: usize,
    burstend: usize,
    linelength: usize,
    lines: usize,
) -> Vec<u16> {
    const STARTING_LINE: usize = 16;
    const SIGNED_SAMPLE_MAX: f32 = 32767.0;
    assert!(lines > STARTING_LINE);

    // Burst-normalize each line and encode it straight to the u16 output in one
    // pass. The lead-in lines below STARTING_LINE (and any samples past `lines`)
    // carry no normalization: the old two-step path left them 0.0f32 and then
    // encoded that to the 32767 zero level, so initialize them to that level and
    // overwrite the normalized region in place. This drops the intermediate
    // field-sized f32 buffer and the separate encode pass that re-read it.
    let zero_level = SIGNED_SAMPLE_MAX as u16;
    let mut output = vec![zero_level; chroma.len()];

    for linenumber in STARTING_LINE..lines {
        let linestart = linelength * linenumber;
        let lineend = linestart + linelength;
        let line = &chroma[linestart..lineend];
        let burst_abs_mean = rms(&line[burststart..burstend]);
        let scale = if burst_abs_mean != 0.0 {
            burst_abs_ref / burst_abs_mean as f32
        } else {
            1.0
        };
        for (out, &sample) in output[linestart..lineend].iter_mut().zip(line.iter()) {
            let scaled = sample * scale;
            *out = ((scaled + SIGNED_SAMPLE_MAX) as i64) as u16;
        }
    }

    output
}

fn upconvert_chroma(
    chroma: &[f32],
    field: &DecodedField,
    chroma_heterodyne: &[Vec<f32>],
) -> Result<Vec<f32>> {
    let lineoffset = field.lineoffset + 1;
    let phase_rotation_sequence = field
        .phase_sequence
        .as_ref()
        .context("missing phase sequence")?;
    let mut uphet = vec![0.0f32; chroma.len()];

    for &(linenumber, current_phase, ..) in phase_rotation_sequence {
        let linestart = (linenumber as isize - lineoffset as isize) * field.outlinelen as isize;
        let lineend = linestart + field.outlinelen as isize;
        let start = resolve_slice_bound(chroma.len(), linestart);
        let end = resolve_slice_bound(chroma.len(), lineend);
        if start >= end {
            continue;
        }

        // Pair the line as slices so the loop carries no per-sample bounds
        // checks and vectorizes.
        let heterodyne_row = &chroma_heterodyne[current_phase][start..end];
        for ((out, &sample), &het) in uphet[start..end]
            .iter_mut()
            .zip(&chroma[start..end])
            .zip(heterodyne_row)
        {
            *out = sample * het;
        }
    }

    Ok(uphet)
}

fn burst_deemphasis(chroma: &mut [f32], field: &DecodedField, burstarea_end: usize) {
    let lineoffset = field.lineoffset + 1;
    for line in lineoffset..field.outlinecount + lineoffset {
        let linestart = (line - lineoffset) * field.outlinelen;
        let lineend = linestart + field.outlinelen;
        for sample in &mut chroma[linestart + burstarea_end + 5..lineend] {
            *sample *= 2.0;
        }
    }
}

fn comb_c(data: &mut [f32], line_len: usize, line_distance: usize) {
    let numlines = data.len() / line_len;
    if numlines <= 2 {
        return;
    }

    // The comb writes each line from its own (current) sample, the line
    // `line_distance` ahead, and the line `line_distance` behind. The current
    // and ahead lines are never overwritten before they are read (lines are
    // processed in increasing order), so they read straight from `data`. Only
    // the behind line may already be overwritten, so keep just the last
    // `line_distance` original lines in a small ring instead of copying the
    // whole field as the previous `data.to_vec()` did.
    let mut ring = vec![0.0f32; line_distance * line_len];
    for line_num in 16..numlines - 2 {
        let line_start = line_num * line_len;
        let advanced_start = (line_num + line_distance) * line_len;
        let slot = (line_num % line_distance) * line_len;
        let delayed = line_num - line_distance;
        // The first `line_distance` lines reach back before the processed range,
        // so those originals are still live in `data`; later ones come from the
        // ring slot they last wrote.
        let delayed_from_data = delayed < 16;
        let delayed_start = delayed * line_len;
        for offset in 0..line_len {
            let i = line_start + offset;
            let current = data[i];
            let advanced = data[advanced_start + offset];
            let delayed_sample = if delayed_from_data {
                data[delayed_start + offset]
            } else {
                ring[slot + offset]
            };
            ring[slot + offset] = current;
            data[i] = (current * 2.0 - advanced - delayed_sample) / 4.0;
        }
    }
}

/// Form the analytic signal of a real SECAM subcarrier while applying the
/// anti-bell HF de-emphasis (§3.4) as a zero-phase magnitude multiply on the
/// positive half-spectrum. This is `hilbert_f32` with an extra per-bin gain:
/// the negative half is zeroed and the positive (interior) half doubled exactly
/// as for the Hilbert transform, and each surviving bin is additionally scaled
/// by `gain[bin]`. With `gain == None` it reduces to `hilbert_f32`. Applying the
/// de-emphasis here (rather than as a causal IIR before the FFT) keeps it
/// zero-phase, so it flattens the transmitted side-component boost without
/// perturbing the subcarrier's instantaneous phase — i.e. the recovered FM.
fn analytic_antibell(
    input: &[f32],
    forward_fft: &dyn Fft<f32>,
    inverse_fft: &dyn Fft<f32>,
    gain: Option<&[f32]>,
) -> Vec<Complex32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let mut spectrum = fft_real_f32(input, forward_fft);
    let g = |i: usize| gain.map_or(1.0, |t| t[i]);
    if n.is_multiple_of(2) {
        spectrum[0] *= g(0);
        for i in 1..(n / 2) {
            spectrum[i] *= 2.0 * g(i);
        }
        spectrum[n / 2] *= g(n / 2);
        for sample in &mut spectrum[(n / 2 + 1)..] {
            *sample = Complex32::new(0.0, 0.0);
        }
    } else if n > 1 {
        spectrum[0] *= g(0);
        for i in 1..=((n - 1) / 2) {
            spectrum[i] *= 2.0 * g(i);
        }
        for sample in &mut spectrum[n.div_ceil(2)..] {
            *sample = Complex32::new(0.0, 0.0);
        }
    } else {
        spectrum[0] *= g(0);
    }
    ifft_complex_owned_f32(spectrum, inverse_fft)
}

/// Median-filter a recovered frequency line over a centred window of `win`
/// samples (clamped at the ends); `win <= 1` returns the input unchanged. FM
/// discrimination produces impulsive "click" spikes wherever the subcarrier
/// envelope momentarily collapses during a fast colour transition; a median
/// rejects those spikes while preserving the transition step itself, which a
/// moving average would instead smear into horizontal colour bleed.
fn median_filter(input: &[f32], win: usize) -> Vec<f32> {
    if win <= 1 || input.len() < 2 {
        return input.to_vec();
    }
    let half = win / 2;
    let n = input.len();
    let mut out = vec![0.0f32; n];
    let mut scratch: Vec<f32> = Vec::with_capacity(win);
    for k in 0..n {
        let lo = k.saturating_sub(half);
        let hi = (k + half + 1).min(n);
        scratch.clear();
        scratch.extend_from_slice(&input[lo..hi]);
        scratch.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        out[k] = scratch[scratch.len() / 2];
    }
    out
}

/// Apply the SECAM LF de-emphasis (§3.2) to one recovered colour-difference
/// baseband line as a single forward pass, which yields the exact `|A_НЧ|⁻¹`
/// magnitude (a zero-phase `filtfilt` would square it). A short constant-valued
/// pre-pad absorbs the filter's start-up transient so the left edge of the
/// active line is not corrupted; the residual constant group delay (~1.2 µs) is
/// immaterial because the luma path is free to be delayed to match (§2).
fn deemphasis_lf(sos: &[Sos<f32>], baseband: &[f32]) -> Vec<f32> {
    const PAD: usize = 64;
    if baseband.is_empty() {
        return Vec::new();
    }
    let mut padded = Vec::with_capacity(baseband.len() + PAD);
    padded.extend(std::iter::repeat(baseband[0]).take(PAD));
    padded.extend_from_slice(baseband);
    let filtered = sosfilt_f32(sos, &padded);
    filtered[PAD..].to_vec()
}

/// Decode a SECAM chroma field (the FM subcarrier, real-valued on the 4·fSC
/// output grid). The subcarrier is FM-demodulated (§5.1) to recover the
/// per-line, alternating `E'_{R-Y}` / `E'_{B-Y}` baseband: a power-weighted
/// cross-product discriminator (steps 3+5) followed by an edge-preserving median
/// that rejects the FM click noise at colour transitions without smearing them.
/// The standard SECAM HF (anti-bell, step 2) and LF (step 6) de-emphasis are
/// applied only when `spec.rf_secam_deemphasis` is set (`--secam-deemphasis`);
/// they are correct only for sources carrying the standard pre-emphasis and are
/// off by default.
///
/// The recovered baseband is then emitted according to `spec.rf_secam_mode`:
/// * [`SecamMode::PseudoPal`] re-modulates it onto the locally-generated
///   `rf_fsc_wave` (quarter-rate = PAL fSC) as a balanced (suppressed-carrier)
///   signal — R-Y lines carry V on the ∓sin axis with the PAL line-switch, B-Y
///   lines carry U on the cos axis — plus a synthesized swinging colour burst,
///   so a standard PAL chroma decoder can demodulate it.
/// * [`SecamMode::RawDemod`] writes the raw demodulated colour difference
///   directly as the chroma signal (no PAL subcarrier), for inspecting the
///   demodulator output.
///
/// `acc` later burst-normalizes the result. Operates in place on `uphet`.
fn process_secam_chroma(uphet: &mut [f32], field: &DecodedField, spec: &DecoderSpec) -> Result<()> {
    let secam = spec
        .rf_secam_params
        .as_ref()
        .context("missing SECAM parameters for chroma decoding")?;
    let raw_demod = matches!(spec.rf_secam_mode, Some(SecamMode::RawDemod));

    // --- Sign/axis conventions (resolved empirically against the reference). ---
    const RB_SWAP: bool = false; // false: the higher rest-freq cluster is R-Y
    const V_SIGN: f32 = 1.0; // sign of the V (R-Y) component
    const U_SIGN: f32 = 1.0; // sign of the U (B-Y) component
    const G_SIGN: f32 = 1.0; // overall PAL line-switch polarity (V-switch + burst)
    const SQRT1_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

    // Invented signal-processing tuning (CLI-controlled, `--secam-*`): a *small*
    // power-weighted discriminator window keeps horizontal sharpness, and an
    // edge-preserving MEDIAN removes the FM click spikes at sharp transitions
    // (where the subcarrier envelope collapses) without the colour bleed a wide
    // average would cause.
    let disc_win = spec.rf_secam_disc_window.max(1);
    let med_win = spec.rf_secam_median_window;

    // SECAM demodulation gains (§5.1) and PAL compression (§4.1), from sys_params.
    let sens_r = secam.sens_ry_hz as f32; // E'_{R-Y} = -Δf / sens_ry
    let sens_b = secam.sens_by_hz as f32; // E'_{B-Y} = +Δf / sens_by
    let compress_v = secam.compress_v; // V = compress_v · E'_{R-Y}
    let compress_u = secam.compress_u; // U = compress_u · E'_{B-Y}
    let a_burst = secam.burst_amplitude; // acc renormalizes; only the
                                         // chroma-to-burst ratio survives.

    let outlinelen = field.outlinelen;
    if outlinelen == 0 || field.outlinecount == 0 {
        return Ok(());
    }
    let phase_sequence = field
        .phase_sequence
        .as_ref()
        .context("missing phase sequence for SECAM chroma decoding")?;
    let lineoffset = field.lineoffset + 1;
    let out_rate_hz = (spec.sys_outfreq * 1e6) as f32;

    // 1) Form the analytic signal, optionally pre-shaped by the anti-bell HF
    // de-emphasis (§3.4 / §5.1 step 2), folded into the same FFT as a zero-phase
    // magnitude multiply. The anti-bell gain is `Some` only in --secam-deemphasis
    // mode (see DecoderSpec); by default it is `None` and this is a plain Hilbert
    // transform. The FM discriminator (below) is the cross-product z[i]·conj(z[i-1])
    // summed over a small window: its angle is the instantaneous frequency. The
    // sum is power-weighted (by envelope²), so it tracks per-sample SNR — this
    // plays the role of the hard-limiter+discriminator pair (§5.1 steps 3+5).
    let mut analytic = analytic_antibell(
        uphet,
        spec.fft_field_forward_f32.as_ref(),
        spec.fft_field_inverse_f32.as_ref(),
        spec.chroma_secam_antibell_gain.as_deref(),
    );
    // Hard-limit (§5.1 step 3) only in de-emphasis mode: the anti-bell magnitude
    // shaping would otherwise bias the power-weighted discriminator, so when it
    // is applied the amplitude is normalised away first.
    if spec.rf_secam_deemphasis {
        for z in analytic.iter_mut() {
            let m = (z.re * z.re + z.im * z.im).sqrt();
            if m > 1e-12 {
                z.re /= m;
                z.im /= m;
            }
        }
    }
    let scale = out_rate_hz / (2.0 * std::f32::consts::PI);
    let n_tot = analytic.len();
    let mut cross_re = vec![0.0f32; n_tot];
    let mut cross_im = vec![0.0f32; n_tot];
    for i in 1..n_tot {
        let (a, b) = (analytic[i].re, analytic[i].im);
        let (c, d) = (analytic[i - 1].re, analytic[i - 1].im);
        cross_re[i] = a * c + b * d; // Re(z[i]·conj(z[i-1]))
        cross_im[i] = b * c - a * d; // Im(z[i]·conj(z[i-1]))
    }
    if n_tot > 1 {
        cross_re[0] = cross_re[1];
        cross_im[0] = cross_im[1];
    }

    // Per-line sample windows. The back-porch protective burst sits at the
    // line's rest frequency and is the primary R/B identifier.
    let (bstart, bend) = padded_burst_area(spec);
    let bp_lo = (bstart + 8).max(0) as usize;
    let bp_hi = (bend - 4).max(bstart) as usize;
    let burst_lo = bstart.max(0) as usize;
    let burst_hi = (bend.max(0) as usize).min(outlinelen);
    let to_sample = |us: f64| ((us * spec.sys_outfreq + BADJ).round() as isize).max(0) as usize;
    let active_lo = to_sample(spec.sys_active_video_us[0]);
    let active_hi = to_sample(spec.sys_active_video_us[1]).min(outlinelen);

    // 2) Classify each line as R-Y / B-Y from its back-porch rest frequency.
    struct LineInfo {
        linestart: usize,
        bp_freq: f32,
    }
    let mut lines: Vec<LineInfo> = Vec::with_capacity(phase_sequence.len());
    for &(linenumber, ..) in phase_sequence {
        let ls = (linenumber as isize - lineoffset as isize) * outlinelen as isize;
        if ls < 0 {
            continue;
        }
        let linestart = ls as usize;
        if bp_hi <= bp_lo || linestart + bp_hi > n_tot {
            continue;
        }
        // Rest frequency from the back-porch protective burst: angle of the
        // summed cross-product (same amplitude-weighted estimator as the active
        // line) is more robust than averaging a per-sample phase derivative.
        let sr: f32 = cross_re[linestart + bp_lo..linestart + bp_hi].iter().sum();
        let si: f32 = cross_im[linestart + bp_lo..linestart + bp_hi].iter().sum();
        let bp_freq = si.atan2(sr) * scale;
        if !bp_freq.is_finite() {
            continue;
        }
        lines.push(LineInfo { linestart, bp_freq });
    }
    if lines.is_empty() {
        return Ok(());
    }
    // Median split into two clusters to bootstrap the high/low rest-frequency
    // centers (used only as a reference for scoring the two phasings below).
    let mut freqs: Vec<f32> = lines.iter().map(|l| l.bp_freq).collect();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = freqs[freqs.len() / 2];
    let (mut hi_sum, mut hi_n, mut lo_sum, mut lo_n) = (0.0f32, 0usize, 0.0f32, 0usize);
    for l in &lines {
        if l.bp_freq > median {
            hi_sum += l.bp_freq;
            hi_n += 1;
        } else {
            lo_sum += l.bp_freq;
            lo_n += 1;
        }
    }
    let hi_center = if hi_n > 0 { hi_sum / hi_n as f32 } else { median };
    let lo_center = if lo_n > 0 { lo_sum / lo_n as f32 } else { median };

    // R-Y and B-Y lines strictly alternate. Rather than classify each line
    // independently from its (noisy) back-porch frequency — where a single
    // misread flips one line and injects a wrong-axis streak — we only let the
    // statistics choose between the two possible phasings. The line type is then
    // locked to the parity of the field-relative line index. Score each phasing
    // by how often the measured back-porch frequency agrees with it.
    let parity = |l: &LineInfo| ((l.linestart / outlinelen) & 1) == 0;
    let mut parity_score = 0i32; // >= 0: even-parity lines are the high (R-Y) cluster
    for l in &lines {
        let measured_high = (l.bp_freq - hi_center).abs() <= (l.bp_freq - lo_center).abs();
        parity_score += if measured_high == parity(l) { 1 } else { -1 };
    }
    let even_is_high = parity_score >= 0;

    // With membership now known exactly from the alternation, set each rest
    // frequency to the median (not the mean) of its parity-assigned group, so a
    // few outlier back-porch readings cannot drag the centre off.
    let mut hi_freqs: Vec<f32> = Vec::new();
    let mut lo_freqs: Vec<f32> = Vec::new();
    for l in &lines {
        if parity(l) == even_is_high {
            hi_freqs.push(l.bp_freq);
        } else {
            lo_freqs.push(l.bp_freq);
        }
    }
    let median_of = |mut v: Vec<f32>, fallback: f32| -> f32 {
        if v.is_empty() {
            return fallback;
        }
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let hi_center = median_of(hi_freqs, hi_center);
    let lo_center = median_of(lo_freqs, lo_center);

    // 3) Rebuild uphet as a pseudo-PAL composite (blanking = 0).
    uphet.iter_mut().for_each(|s| *s = 0.0);
    // LF de-emphasis is `Some` only when --secam-deemphasis is set.
    let lf_deemph = spec.chroma_secam_lf_deemphasis.as_ref();
    let lpf = spec.chroma_secam_baseband_lpf.as_ref();
    let mut baseband = vec![0.0f32; active_hi.saturating_sub(active_lo)];
    let mut freq = vec![0.0f32; active_hi.saturating_sub(active_lo)];

    for l in &lines {
        // Type comes from the alternation (parity), not a per-line measurement.
        let is_high = parity(l) == even_is_high;
        let is_ry = is_high ^ RB_SWAP; // higher rest freq (4.40625) == R-Y
        let rest = if is_high { hi_center } else { lo_center };
        let g = G_SIGN * if is_ry { 1.0 } else { -1.0 };
        let base = l.linestart;

        // Recover the baseband colour-difference signal over the active line.
        if active_hi > active_lo && base + active_hi <= n_tot {
            let n = active_hi - active_lo;
            // Prefix sums of the cross-product over the active line, for an O(n)
            // moving-window discriminator. The window is bounded to the active
            // region so blanking samples never leak into the estimate.
            let mut pre_re = vec![0.0f32; n + 1];
            let mut pre_im = vec![0.0f32; n + 1];
            for k in 0..n {
                pre_re[k + 1] = pre_re[k] + cross_re[base + active_lo + k];
                pre_im[k + 1] = pre_im[k] + cross_im[base + active_lo + k];
            }
            let half = disc_win / 2;
            for k in 0..n {
                let lo = k.saturating_sub(half);
                let hi = (k + half + 1).min(n);
                let sr = pre_re[hi] - pre_re[lo];
                let si = pre_im[hi] - pre_im[lo];
                freq[k] = si.atan2(sr) * scale;
            }
            // Edge-preserving spike rejection, then descale the deviation to the
            // recovered colour difference E' (the PAL compression is a modulation
            // step applied below, so RawDemod sees the uncompressed signal).
            let freq_med = median_filter(&freq[..n], med_win);
            for k in 0..n {
                let dev = freq_med[k] - rest;
                baseband[k] = if is_ry {
                    V_SIGN * (-dev / sens_r)
                } else {
                    U_SIGN * (dev / sens_b)
                };
            }
            // LF de-emphasis (§5.1 step 6), then band-limit to the PAL chroma
            // bandwidth (§5.2 step 1).
            let deemphasized;
            let pre: &[f32] = match lf_deemph {
                Some(sos) => {
                    deemphasized = deemphasis_lf(sos, &baseband);
                    &deemphasized
                }
                None => &baseband,
            };
            let filtered;
            let bb: &[f32] = match lpf {
                Some(sos) => {
                    filtered = sosfiltfilt_f32(sos, pre);
                    &filtered
                }
                None => pre,
            };
            for (k, col) in (active_lo..active_hi).enumerate() {
                let i = base + col;
                uphet[i] = if raw_demod {
                    // Raw demodulated colour difference, no PAL modulation.
                    bb[k]
                } else {
                    // Pseudo-PAL: compress (§4.1) and quadrature-modulate.
                    let (sin_i, cos_i) = spec.rf_fsc_wave[i];
                    if is_ry {
                        -g * compress_v * bb[k] * sin_i
                    } else {
                        compress_u * bb[k] * cos_i
                    }
                };
            }
        }

        // Synthesize the PAL swinging burst (135° when g=+1, 225° when g=-1).
        // Emitted in both modes so `acc` has a stable reference to normalize by;
        // in RawDemod it is just a back-porch reference, not part of the picture.
        if base + burst_hi <= uphet.len() {
            for col in burst_lo..burst_hi {
                let i = base + col;
                let (sin_i, cos_i) = spec.rf_fsc_wave[i];
                uphet[i] = SQRT1_2 * a_burst * (-cos_i - g * sin_i);
            }
        }
    }

    Ok(())
}

fn process_chroma_internal(
    field: &mut DecodedField,
    spec: &DecoderSpec,
    chroma_afc_state: &mut ChromaAfcState,
) -> Result<Vec<u16>> {
    let chroma_downscaled = downscale_raw_vec(field, None, None, None, true)?;
    let mut chroma: Vec<f32> = if spec.chroma_afc_enabled() {
        let bandpass = chroma_afc_state.get_chroma_bandpass(spec)?;
        let chroma_len = chroma_downscaled.len();
        let chroma = demod_chroma_filt_array(
            &chroma_downscaled,
            spec,
            &bandpass,
            chroma_len,
            Some((10.0 * (spec.sys_outfreq / 40.0)) as isize),
        );
        chroma_afc_state.freq_offset(spec, &chroma, true)?;
        chroma
    } else {
        chroma_downscaled
    };
    let burstarea = padded_burst_area(spec);

    let is_ntsc = spec.color_system == ColorSystem::Ntsc;
    if is_ntsc {
        burst_deemphasis(&mut chroma, field, burstarea.1 as usize)
    }

    let chroma_heterodyne = active_chroma_heterodyne(spec, chroma_afc_state);

    let mut uphet = upconvert_chroma(&chroma, field, chroma_heterodyne)?;

    if is_ntsc && !spec.rf_disable_phase_correction {
        // Rotate the chroma so the measured burst phase lines up with 0 degrees.
        let burst_phase_avg = field.burst_phase_avg.context("missing burst phase avg")?;
        let hilbert = hilbert_f32(
            &uphet,
            spec.fft_field_forward_f32.as_ref(),
            spec.fft_field_inverse_f32.as_ref(),
        );
        adjust_phase(&hilbert, &mut uphet, burst_phase_avg, 0.0);
    }

    uphet = sosfiltfilt_f32(&spec.chroma_filter_final, &uphet);

    if let Some(sos) = spec.chroma_filter_deemphasis.as_ref() {
        uphet = sosfilt_f32(sos, &uphet);
    }

    if !spec.rf_disable_comb {
        let line_distance = if is_ntsc { 1 } else { 2 };
        comb_c(&mut uphet, field.outlinelen, line_distance);
    }

    if spec.rf_secam_mode.is_some() {
        process_secam_chroma(&mut uphet, field, spec)?;
    }

    let burst_abs_ref = spec.sys_burst_abs_ref.context("missing burst_abs_ref")?;
    Ok(acc(
        &uphet,
        burst_abs_ref,
        burstarea.0 as usize,
        burstarea.1 as usize,
        field.outlinelen,
        field.outlinecount,
    ))
}

pub(crate) fn decode_chroma(
    field: &mut DecodedField,
    spec: &DecoderSpec,
    chroma_afc_state: &mut ChromaAfcState,
) -> Result<Option<Vec<u16>>> {
    if !spec.rf_write_chroma || spec.color_system == ColorSystem::Monochrome {
        return Ok(None);
    }
    let upconverted = process_chroma_internal(field, spec, chroma_afc_state)?;
    Ok(Some(upconverted))
}
