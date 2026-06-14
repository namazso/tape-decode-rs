use anyhow::{bail, Error, Result};
use serde::Deserialize;

/// Shelving filter orientation used by custom profile filters and [`crate::spec::gen_shelf`].
#[derive(Clone, Copy, Debug, Deserialize)]
pub enum ShelfKind {
    Low,
    High,
}

/// How to react when two consecutive fields share the same field order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
pub enum FieldOrderAction {
    Detect,
    Duplicate,
    Drop,
    None,
}

/// How SECAM chroma is handled when `--secam` is given (otherwise the chroma is
/// left untouched).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
pub enum SecamMode {
    /// FM-demodulate the SECAM subcarrier and re-modulate the recovered
    /// colour-difference signals as a balanced (suppressed-carrier) pseudo-PAL
    /// composite that a standard PAL chroma decoder can demodulate.
    PseudoPal,
    /// Emit the raw demodulated, line-alternating colour-difference baseband
    /// (Dr/Db) directly as the chroma signal, without re-modulating it onto a
    /// PAL subcarrier. For inspecting the demodulator output.
    RawDemod,
}

/// SECAM demodulation/modulation constants drawn from the SECAM/PAL standards
/// (see `tmp/secam-to-pal.md`). Carried per-system in `sys_params.json`; present
/// only for SECAM systems.
#[derive(Clone, Debug, Deserialize)]
pub struct SecamParams {
    /// R-Y FM sensitivity (Hz): `E'_{R-Y} = -Δf / sens_ry_hz` (§5.1, 532 kHz).
    pub sens_ry_hz: f64,
    /// B-Y FM sensitivity (Hz): `E'_{B-Y} = +Δf / sens_by_hz` (§5.1, 345 kHz).
    pub sens_by_hz: f64,
    /// PAL compression of V (R-Y): `V = compress_v · E'_{R-Y}` (§4.1, 0.877).
    pub compress_v: f32,
    /// PAL compression of U (B-Y): `U = compress_u · E'_{B-Y}` (§4.1, 0.493).
    pub compress_u: f32,
    /// Pseudo-PAL swinging-burst amplitude (§4.5, 0.15). Only the
    /// chroma-to-burst ratio survives the later burst normalization.
    pub burst_amplitude: f32,
    /// Recovered-baseband low-pass cutoff (Hz): PAL chroma bandwidth (§4.1, 1.3 MHz).
    pub baseband_lpf_hz: f64,
    /// Anti-bell (HF de-emphasis) resonator centre (Hz) (§3.4, 4.286 MHz).
    pub bell_centre_hz: f64,
    /// Anti-bell resonator quality factor (§3.4, 16).
    pub bell_q: f64,
    /// LF de-emphasis first-order pole (Hz) (§3.2, 85 kHz); zero at `k·pole`.
    pub lf_deemph_pole_hz: f64,
    /// LF de-emphasis zero/pole ratio `k` (§3.2, 3).
    pub lf_deemph_k: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct DecodeOptions {
    pub cafc: bool,
    pub chroma_afc_fine_tune_fh_ratio: f64,
    pub fallback_vsync: bool,
    pub field_order_action: FieldOrderAction,
    pub fm_audio_notch: f64,
    pub use_fsc_notch_filter: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            cafc: false,
            chroma_afc_fine_tune_fh_ratio: 0.25,
            fallback_vsync: false,
            field_order_action: FieldOrderAction::Detect,
            fm_audio_notch: 0.0,
            use_fsc_notch_filter: false,
        }
    }
}

/// Interpolation used by the wow level adjustment, mapped to a spline degree.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
pub enum WowInterpolation {
    Linear,
    Quadratic,
    Cubic,
}

impl WowInterpolation {
    pub fn spline_degree(self) -> usize {
        match self {
            WowInterpolation::Linear => 1,
            WowInterpolation::Quadratic => 2,
            WowInterpolation::Cubic => 3,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct DecodeProfile {
    pub sys_params: SysParams,
    pub decoder_params: DecoderParams,
    #[serde(default)]
    pub decode_options: DecodeOptions,
}

/// Scanning standard, defined purely by line count. Luma decoding depends only
/// on this.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
#[serde(try_from = "usize")]
pub enum LineSystem {
    Line405 = 405,
    Line819 = 819,
    Line525 = 525,
    Line625 = 625,
}

impl TryFrom<usize> for LineSystem {
    type Error = Error;

    fn try_from(frame_lines: usize) -> Result<Self> {
        match frame_lines {
            405 => Ok(LineSystem::Line405),
            819 => Ok(LineSystem::Line819),
            525 => Ok(LineSystem::Line525),
            625 => Ok(LineSystem::Line625),
            other => bail!("unsupported line count: {other}"),
        }
    }
}

impl LineSystem {
    /// Number of lines per frame, i.e. the variant's discriminant value.
    pub fn line_count(self) -> usize {
        self as usize
    }
}

/// Colour encoding standard configured by resolved system parameters. Profiles
/// with no chroma processing use [`ColorSystem::Monochrome`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
pub enum ColorSystem {
    #[serde(rename = "NTSC")]
    Ntsc,
    #[serde(rename = "PAL")]
    Pal,
    #[serde(rename = "SECAM")]
    Secam,
    Monochrome,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SysParams {
    pub color_system: ColorSystem,
    pub fsc_mhz: f64,
    pub frame_lines: LineSystem,
    pub field_lines: [i64; 2],
    pub line_period: f64,
    pub active_video_us: [f64; 2],
    pub fps: f64,
    pub ire0: f32,
    pub hz_ire: f32,
    pub vsync_ire: f32,
    pub color_burst_us: [f64; 2],
    pub blacksnr_slice: [usize; 3],

    pub num_pulses: usize,
    pub hsync_pulse_us: f64,
    pub eq_pulse_us: f64,
    pub vsync_pulse_us: f64,
    pub output_zero: i64,
    pub outlinelen: usize,
    pub outfreq: f64,
    pub ld_vits_whitelocs: Vec<[usize; 3]>,
    pub burst_abs_ref: Option<f32>,
    pub track_ire0_offset: [f64; 2],
    pub nonlinear_deviation: Option<f32>,
    /// SECAM demodulation/modulation constants; present only for SECAM systems.
    #[serde(default)]
    pub secam: Option<SecamParams>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DecoderParams {
    pub video_bpf: Option<VideoBpf>,
    pub video_lpf_freq: f64,
    pub video_lpf_order: usize,

    pub deemph: Option<DeemphasisParams>,
    pub nonlinear: NonlinearParams,
    pub video_lpf_extra: f64,
    pub video_lpf_extra_order: usize,
    pub video_hpf_extra: f64,
    pub video_hpf_extra_order: usize,
    pub video_lpf_supergauss: bool,
    pub video_custom_luma_filters: Option<Vec<VideoLumaFilter>>,
    pub video_rf_peak: Option<RfPeaking>,
    pub video_eq: Option<VideoEqParams>,
    /// Force color system to [`ColorSystem::Monochrome`] to prevent processing.
    #[serde(default)]
    pub is_composite_color: bool,
    pub color_under_carrier: f64,
    /// When set, the color-under chroma is run through the deemphasis filter
    /// during upconversion. Has no effect on monochrome profiles.
    #[serde(default)]
    pub chroma_deemphasis_enabled: bool,
    pub chroma_bpf_upper: f64,
    pub chroma_bpf_order: usize,
    pub chroma_bpf_lower: f64,
    pub chroma_rotation: Option<[i64; 2]>,
    pub chroma_audio_notch_freq: f64,
    pub chroma_offset: f64,
    pub fm_audio_channels: Option<FmAudioChannels>,
    pub boost_bpf: Option<BoostBpf>,
    pub boost_ramp: Option<BoostRampFilter>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VideoEqBand {
    pub corner: f64,
    pub transition: f64,
    pub order_limit: usize,
}
#[derive(Clone, Debug, Deserialize)]
pub struct VideoEqParams {
    pub loband: VideoEqBand,
}
#[derive(Clone, Debug, Deserialize)]
pub struct FmAudioChannels {
    pub channel_0_freq: f64,
    pub channel_1_freq: f64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct RfPeaking {
    pub freq: f64,
    pub gain: f64,
    pub bandwidth: f64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct NotchFilter {
    pub freq: f64,
    pub q: f64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct BoostRampFilter {
    pub rf_linear_0: f64,
    pub rf_linear_20: f64,
    pub start_freq: f64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct VideoBpf {
    pub low: f64,
    pub high: f64,
    pub order: usize,
}
#[derive(Clone, Debug, Deserialize)]
pub struct BoostBpf {
    pub low: f64,
    pub high: f64,
    pub mult: f64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct DeemphasisParams {
    pub mid: f64,
    pub gain: f64,
    pub q: f64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct LogisticParams {
    pub mid: f32,
    pub rate: f32,
}
#[derive(Clone, Debug, Deserialize)]
pub struct NonlinearParams {
    pub highpass_freq: f64,
    pub highpass_limit_h: f32,
    pub highpass_limit_l: f32,
    pub exp_scaling: f32,
    pub scaling_1: Option<f32>,
    pub scaling_2: Option<f32>,
    pub logistic: Option<LogisticParams>,
    pub static_factor: Option<f32>,
    pub bandpass_upper: Option<f64>,
    pub bandpass_order: usize,
    pub amp_lpf_freq: f64,
    pub use_sub_deemphasis: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum VideoLumaFilter {
    File {
        filename: String,
    },
    Shelf {
        shelf_kind: ShelfKind,
        gain: f64,
        midfreq: f64,
        q: f64,
    },
}

fn default_secam_disc_window() -> usize {
    5
}

fn default_secam_median_window() -> usize {
    27
}

#[derive(Clone, Debug, Deserialize)]
pub struct DecodeRequest {
    pub inputfreq: f64,

    // RF options. Fields that the DecoderSpec constructor copies through
    // unchanged carry the DecoderSpec member name.
    pub chroma_trap: bool,
    pub sharpness: i64,
    pub notch: Option<NotchFilter>,
    pub dod_threshold_p: f32,
    pub dod_threshold_a: Option<f32>,
    pub dod_hysteresis: f32,
    pub track_phase: Option<i64>,
    pub high_boost: Option<f64>,
    pub video_disable_diff_demod: bool,
    pub fm_audio_notch: f64,
    pub rf_disable_dc_offset: bool,
    pub disable_comb: bool,
    /// SECAM chroma handling. `None` leaves the chroma untouched.
    #[serde(default)]
    pub secam: Option<SecamMode>,
    /// Apply the SECAM HF (anti-bell, §3.4) and LF (§3.2) de-emphasis during the
    /// SECAM conversion. Only meaningful when `secam` is set. Off by default: it
    /// is correct only for sources that carry the standard SECAM pre-emphasis,
    /// and degrades sources (e.g. many test patterns) that do not.
    #[serde(default)]
    pub secam_deemphasis: bool,
    /// FM-discriminator averaging window (samples) for the SECAM conversion;
    /// smaller is sharper. An invented tuning parameter, not from the standard.
    #[serde(default = "default_secam_disc_window")]
    pub secam_disc_window: usize,
    /// Median window (samples) that rejects FM click noise at colour transitions
    /// during the SECAM conversion. An invented tuning parameter.
    #[serde(default = "default_secam_median_window")]
    pub secam_median_window: usize,
    pub skip_chroma: bool,
    pub video_nldeemp_enabled: bool,
    pub subdeemp: bool,
    pub y_comb: f32,
    pub cafc: bool,
    pub rf_disable_right_hsync: bool,
    pub level_detect_divisor: i64,
    pub fallback_vsync: bool,
    pub rf_relaxed_line0: bool,
    pub rf_field_order_confidence: i64,
    pub rf_saved_levels: bool,
    pub rf_skip_hsync_refine: bool,
    pub rf_export_raw_tbc: bool,
    pub rf_ire0_adjust: bool,
    pub rf_detect_chroma_track_phase: bool,
    pub rf_disable_burst_hsync: bool,
    pub rf_disable_phase_correction: bool,

    // Extra options.
    pub wow_level_adjust_smoothing: Option<f32>,
    pub wow_interpolation_method: WowInterpolation,

    pub field_order_action: FieldOrderAction,
    pub level_adjust: f32,
    pub do_dod: bool,
    pub decode_profile: DecodeProfile,
    pub ntscj: bool,
}
