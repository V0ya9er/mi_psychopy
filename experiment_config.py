from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExperimentAbort(Exception):
    """Raised when the operator presses ESC to stop the experiment."""


@dataclass(frozen=True)
class TimingConfig:
    fixation_s: float
    cue_s: float
    imagery_s: float
    ao_prime_s: float
    ao_mi_s: float
    mi_only_s: float
    iti_s: float


@dataclass(frozen=True)
class DisplayConfig:
    fullscreen: bool
    window_width: int
    window_height: int
    background_color: str = "black"
    window_size_presets: tuple[tuple[int, int], ...] = ((960, 540), (1280, 720), (1600, 900))
    window_style: str = "psychopy"
    display_index: int = 0
    refresh_rate_hz: float = 60.0


@dataclass(frozen=True)
class StimulusConfig:
    cue_image_path: Path | None
    pure_mi_cue_left_image_path: Path | None
    pure_mi_cue_right_image_path: Path | None
    pure_mi_left_image_path: Path | None
    pure_mi_right_image_path: Path | None
    pure_mi_rest_image_path: Path | None
    ao_left_image_path: Path | None
    ao_right_image_path: Path | None
    ao_video_path: Path | None
    ao_video_start_s: float = 2.0
    ssvep_left_clean_image_path: Path | None = None
    ssvep_right_clean_image_path: Path | None = None
    image_height: float = 0.34
    task_image_scale: float = 1.55


@dataclass(frozen=True)
class SSVEPConfig:
    enabled: bool
    left_freq_hz: float
    right_freq_hz: float
    flicker_duration_s: float
    allow_gaze_shift: bool
    flicker_size: tuple[float, float]
    flicker_y_pos: float
    left_x_pos: float
    right_x_pos: float
    flicker_mode: str = "border"  # "border" = border flicker + static image; "image" = clean image opacity flicker
    display_mode: str = "single_side"  # "both_sides" / "single_center" / "single_side"
    bright_color: str = "white"
    dark_color: str = "black"
    flicker_border_width: float = 4.0
    dim_opacity: float = 0.0  # OFF-state opacity for "image" flicker mode
    waveform: str = "square"  # "square" = square wave (stronger SSVEP) / "sine" = sinusoidal (more comfortable)
    target_ring_color: str = "yellow"
    target_ring_width: float = 5.0
    cue_target_border_color: str = "white"
    cue_nontarget_border_color: str = "#808080"
    task_nontarget_border_color: str = "#666666"


@dataclass(frozen=True)
class P300Config:
    enabled: bool
    task_duration_s: float
    soa_s: float
    flash_duration_s: float
    flash_sequence_seed: int
    left_x_pos: float
    right_x_pos: float
    y_pos: float
    image_size: tuple[float, float]
    target_probability: float = 0.25
    flash_mode: str = "border"  # "border" = border flash + static image; "image" = clean image opacity flash
    flash_border_width: float = 6.0
    flash_color: str = "white"
    noflash_color: str = "#333333"
    dim_opacity: float = 0.0  # OFF-state opacity for "image" flash mode
    target_ring_color: str = "yellow"
    target_ring_width: float = 5.0
    cue_target_border_color: str = "white"
    cue_nontarget_border_color: str = "#808080"


@dataclass(frozen=True)
class ArrowConfig:
    enabled: bool
    arrow_style: str = "unicode"
    arrow_color: str = "white"
    arrow_height: float = 0.20


@dataclass(frozen=True)
class SSVEPArousalConfig:
    enabled: bool
    freq_mode: str = "fixed"            # "fixed" / "random"
    fixed_freq_hz: float = 20.0       # fixed mode frequency
    freq_min_hz: float = 18.0         # random mode min
    freq_max_hz: float = 25.0         # random mode max
    waveform: str = "sine"            # "square" / "sine"
    cue_style: str = "arrow"          # "arrow" / "image"
    task_style: str = "arrow"          # "arrow" / "image"
    flicker_duration_s: float = 4.5    # task duration in seconds
    stimulus_size: tuple[float, float] = (0.35, 0.35)  # (width, height) in PsychoPy height units
    dim_opacity: float = 0.0          # OFF-state opacity for image flicker mode
    arrow_color: str = "white"         # arrow color
    arrow_height: float = 0.20        # arrow size in PsychoPy height units


@dataclass(frozen=True)
class SSVEPSerialConfig:
    enabled: bool
    cue_ssvep_freq_left_hz: float = 10.0    # cue 段左条件 SSVEP 频率
    cue_ssvep_freq_right_hz: float = 15.0   # cue 段右条件 SSVEP 频率
    cue_ssvep_mode: str = "frequency_coded"  # "frequency_coded" (频率编码方向) / "same_freq" (相同频率不编码方向)
    same_freq_hz: float = 20.0 # same_freq 模式下的频率
    cue_ssvep_duration_s: float = 2.0 # cue 段 SSVEP 持续时间
    gap_duration_s: float = 2.0 # 间隔持续时间
    mi_duration_s: float = 4.0 # pure MI 任务持续时间
    waveform: str = "sine" # "square" / "sine"
    cue_style: str = "arrow" # "arrow" / "image" cue 样式
    task_style: str = "arrow" # "arrow" / "image" task 样式
    display_mode: str = "single_center" # "single_center" / "both_sides" 显示模式
    stimulus_width: float = 0.35 # 刺激宽度
    stimulus_height: float = 0.35 # 刺激高度
    border_width: float = 4.0 # 边框宽度
    dim_opacity: float = 0.0 # OFF 态 opacity
    arrow_color: str = "white" # 箭头颜色
    arrow_height: float = 0.20 # 箭头高度


@dataclass(frozen=True)
class SSVEPRTConfig:
    enabled: bool
    mi_enabled: bool = False           # Enable MI model fusion (default: FBCCA only)
    mi_checkpoint_path: str = (
        r"D:\CSDIY\EEG\OLM\mi_benchmark\results"
        r"\0418_eegnet_deepconvnet_shallowconvnet_mi_ssvep_branchnet_mi_ssvep_branchnet_mi_only_mi_ssvep_branchnet_ssvep_only_logreg_svm_random_forest_fbcsp_lda"
        r"\deepconvnet\artifacts\fold_2.pth"
    )  # default: best DeepConvNet fold (99.4% val accuracy)
    classifier_window_s: float = 1.5      # classification window size (1.5s for robust CCA)
    classifier_stride_s: float = 0.25     # sliding window stride
    confidence_threshold: float = 0.15    # minimum confidence for feedback (margin-based)
    # Inherit SSVEP params from SSVEPConfig
    left_freq_hz: float = 10.0
    right_freq_hz: float = 15.0
    flicker_duration_s: float = 4.5
    flicker_mode: str = "border"
    display_mode: str = "single_side"
    waveform: str = "square"
    flicker_size: tuple[float, float] = (0.34, 0.34)
    flicker_y_pos: float = 0.0
    left_x_pos: float = -0.35
    right_x_pos: float = 0.35
    bright_color: str = "white"
    dark_color: str = "black"
    flicker_border_width: float = 4.0
    dim_opacity: float = 0.0
    enable_diag: bool = False          # Enable LSL diagnostics (gap/jitter detection, window saving)
    diag_dir: str = "diag"             # Diagnostic output directory


@dataclass(frozen=True)
class NetworkConfig:
    udp_ip: str
    udp_port: int


@dataclass(frozen=True)
class LabRecorderConfig:
    cli_path: Path
    study_root: str
    path_template: str
    auto_record: bool = True
    stream_queries: tuple[str, ...] = ('name="obci_eeg1"', 'name="obci_eeg2"')
    recording_format: str = "xdf"       # "xdf" / "csv" / "both"
    csv_marker_mode: str = "legacy"     # "legacy" (0=other, 1=left, 2=right) / "detailed" (actual marker values)


@dataclass(frozen=True)
class SessionConfig:
    mode: str
    class_mode: str
    trial_mode: str
    block_count: int
    repeats_per_class: int
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    participant: str
    session: str
    run: int
    timings: TimingConfig
    display: DisplayConfig
    stimuli: StimulusConfig
    ssvep: SSVEPConfig
    p300: P300Config
    arrow: ArrowConfig
    ssvep_arousal: SSVEPArousalConfig
    ssvep_serial: SSVEPSerialConfig
    ssvep_rt: SSVEPRTConfig
    network: NetworkConfig
    labrecorder: LabRecorderConfig
    session_cfg: SessionConfig
    output_dir: Path


@dataclass(frozen=True)
class PhaseSpec:
    phase_name: str
    duration_s: float
    screen_kind: str = "text"
    title: str = ""
    body: str = ""
    footer: str = ""
    layout: str = "stimulus"
    image_path: Path | None = None
    video_path: Path | None = None
    video_start_s: float = 0.0
    video_flip_horizontal: bool = False
    image_scale: float = 1.0
    marker_name: str = ""
    marker_value: int | None = None
    note: str = ""
    center_mode: str = "none"
    ssvep_target_side: str = ""
    ssvep_target_freq_hz: float = 0.0
    ssvep_left_freq_hz: float = 0.0
    ssvep_right_freq_hz: float = 0.0
    p300_target_side: str = ""
    p300_soa_s: float = 0.0
    p300_flash_duration_s: float = 0.0
    ssvep_arousal_freq_hz: float = 0.0


@dataclass(frozen=True)
class Trial:
    block_index: int
    trial_index_in_block: int
    global_trial_index: int
    condition: str
    trial_type: str
    phase_sequence: tuple[str, ...]
    ssvep_target_side: str = ""
    ssvep_target_freq_hz: float = 0.0
    p300_target_side: str = ""
    ssvep_arousal_freq_hz: float = 0.0
