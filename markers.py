MARKERS = {
    "fixation_on": 5,
    "block_start": 90,
    "cue_left": 11,
    "cue_right": 12,
    "cue_rest": 13,
    "mi_left": 21,
    "mi_right": 22,
    "mi_rest": 23,
    "ao_prime_left": 31,
    "ao_prime_right": 32,
    "ao_mi_left": 41,
    "ao_mi_right": 42,
    "mi_only_left": 51,
    "mi_only_right": 52,
    "ssvep_left": 61,
    "ssvep_right": 62,
    "ssvep_gaze_left": 71,
    "ssvep_gaze_right": 72,
    "p300_target_flash": 81,
    "p300_nontarget_flash": 82,
    "task_off": 29,
    "block_end": 91,
    "session_end": 99,
    # 方案4: mi_arrow (纯箭头 MI baseline)
    "arrow_cue_left": 101,
    "arrow_cue_right": 102,
    "arrow_mi_left": 111,
    "arrow_mi_right": 112,
    # 方案2: mi_ssvep_arousal (中央 SSVEP arousal)
    "arousal_cue_left": 121,
    "arousal_cue_right": 122,
    "arousal_task_left": 131,
    "arousal_task_right": 132,
    # 方案1: mi_ssvep_serial (串行 SSVEP→MI)
    "serial_ssvep_cue_left": 141,
    "serial_ssvep_cue_right": 142,
    "serial_gap": 145,
    "serial_mi_left": 151,
    "serial_mi_right": 152,
    # 方案3: mi_audio_fb (听觉 ERD 反馈)
    "audio_fb_cue_left": 161,
    "audio_fb_cue_right": 162,
    "audio_fb_task_left": 171,
    "audio_fb_task_right": 172,
    # 方案5: mi_ssvep_rt (实时分类 SSVEP+MI)
    "rt_ssvep_left": 181,
    "rt_ssvep_right": 182,
    "rt_task_off": 189,
}

ALL_CONDITIONS = ("left", "right", "rest")
CLASS_MODE_TO_CONDITIONS = {
    "binary": ("left", "right"),
    "ternary": ALL_CONDITIONS,
}
TRIAL_MODE_TO_TYPES = {
    "pure_mi": ("pure_mi",),
    "ao_mi": ("ao_mi",),
    "mi_ssvep": ("mi_ssvep",),
    "pure_ssvep": ("pure_ssvep",),
    "mi_p300": ("mi_p300",),
    "mixed": ("pure_mi", "ao_mi"),
    "mi_arrow": ("mi_arrow",),
    "mi_ssvep_arousal": ("mi_ssvep_arousal",),
    "mi_ssvep_serial": ("mi_ssvep_serial",),
    "mi_audio_fb": ("mi_audio_fb",),
    "mi_ssvep_rt": ("mi_ssvep_rt",),
}
TRIAL_TYPE_TO_LABEL = {
    "pure_mi": "纯 MI",
    "ao_mi": "AO+MI",
    "mi_ssvep": "MI+SSVEP",
    "pure_ssvep": "纯 SSVEP",
    "mi_p300": "MI+P300",
    "mi_arrow": "纯箭头 MI",
    "mi_ssvep_arousal": "SSVEP 觉醒增强",
    "mi_ssvep_serial": "串行 SSVEP→MI",
    "mi_audio_fb": "听觉 ERD 反馈",
    "mi_ssvep_rt": "MI+SSVEP 实时分类",
}

CONDITION_TO_CUE_MARKER = {
    "left": MARKERS["cue_left"],
    "right": MARKERS["cue_right"],
    "rest": MARKERS["cue_rest"],
}

CONDITION_TO_MI_MARKER = {
    "left": MARKERS["mi_left"],
    "right": MARKERS["mi_right"],
    "rest": MARKERS["mi_rest"],
}

CONDITION_TO_AO_PRIME_MARKER = {
    "left": MARKERS["ao_prime_left"],
    "right": MARKERS["ao_prime_right"],
}

CONDITION_TO_AO_MI_MARKER = {
    "left": MARKERS["ao_mi_left"],
    "right": MARKERS["ao_mi_right"],
}

CONDITION_TO_MI_ONLY_MARKER = {
    "left": MARKERS["mi_only_left"],
    "right": MARKERS["mi_only_right"],
}

CONDITION_TO_P300_TARGET_MARKER = {
    "left": MARKERS["p300_target_flash"],
    "right": MARKERS["p300_target_flash"],
}

CONDITION_TO_P300_NONTARGET_MARKER = {
    "left": MARKERS["p300_nontarget_flash"],
    "right": MARKERS["p300_nontarget_flash"],
}

CONDITION_TO_CUE_TEXT = {
    "left": "准备左手想象",
    "right": "准备右手想象",
    "rest": "放松身体，准备静息",
}

CONDITION_TO_TASK_TEXT = {
    "left": "开始左手想象",
    "right": "开始右手想象",
    "rest": "开始静息，无需进行任何想象",
}

CONDITION_TO_AO_PRIME_TEXT = {
    "left": "先观察左手动作",
    "right": "先观察右手动作",
}

CONDITION_TO_AO_MI_TEXT = {
    "left": "边观察边做左手想象",
    "right": "边观察边做右手想象",
}

CONDITION_TO_MI_ONLY_TEXT = {
    "left": "继续保持左手运动想象",
    "right": "继续保持右手运动想象",
}

CONDITION_TO_P300_TASK_TEXT = {
    "left": "关注左侧闪光，保持左手运动想象",
    "right": "关注右侧闪光，保持右手运动想象",
}

# New paradigms — marker mappings (text mappings reuse CONDITION_TO_CUE_TEXT / CONDITION_TO_TASK_TEXT)
CONDITION_TO_ARROW_CUE_MARKER = {
    "left": MARKERS["arrow_cue_left"],
    "right": MARKERS["arrow_cue_right"],
}
CONDITION_TO_ARROW_MI_MARKER = {
    "left": MARKERS["arrow_mi_left"],
    "right": MARKERS["arrow_mi_right"],
}
CONDITION_TO_ARROW_CUE_TEXT = CONDITION_TO_CUE_TEXT
CONDITION_TO_ARROW_MI_TEXT = CONDITION_TO_TASK_TEXT

CONDITION_TO_AROUSAL_CUE_MARKER = {
    "left": MARKERS["arousal_cue_left"],
    "right": MARKERS["arousal_cue_right"],
}
CONDITION_TO_AROUSAL_TASK_MARKER = {
    "left": MARKERS["arousal_task_left"],
    "right": MARKERS["arousal_task_right"],
}
CONDITION_TO_AROUSAL_CUE_TEXT = CONDITION_TO_CUE_TEXT
CONDITION_TO_AROUSAL_TASK_TEXT = CONDITION_TO_TASK_TEXT

CONDITION_TO_SERIAL_SSVEP_CUE_MARKER = {
    "left": MARKERS["serial_ssvep_cue_left"],
    "right": MARKERS["serial_ssvep_cue_right"],
}
CONDITION_TO_SERIAL_MI_MARKER = {
    "left": MARKERS["serial_mi_left"],
    "right": MARKERS["serial_mi_right"],
}
CONDITION_TO_SERIAL_SSVEP_CUE_TEXT = CONDITION_TO_CUE_TEXT
CONDITION_TO_SERIAL_MI_TEXT = CONDITION_TO_TASK_TEXT

CONDITION_TO_AUDIO_FB_CUE_MARKER = {
    "left": MARKERS["audio_fb_cue_left"],
    "right": MARKERS["audio_fb_cue_right"],
}
CONDITION_TO_AUDIO_FB_TASK_MARKER = {
    "left": MARKERS["audio_fb_task_left"],
    "right": MARKERS["audio_fb_task_right"],
}

CONDITION_TO_RT_SSVEP_MARKER = {
    "left": MARKERS["rt_ssvep_left"],
    "right": MARKERS["rt_ssvep_right"],
}
CONDITION_TO_RT_SSVEP_TEXT = CONDITION_TO_TASK_TEXT  # reuse existing task text
