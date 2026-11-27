"""
Paket Core Smart Shoe Monitor.
Menyediakan modul konfigurasi terpusat, watcher pergerakan, dan pipeline inferensi AI.
"""
from core.config import (
    BASE_DIR,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_UNCERTAIN_LABEL,
    DEFAULT_AUDIO_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_STREAMLIT_HOST,
    DEFAULT_STREAMLIT_PORT,
    ENSEMBLE_ENABLED,
    FLOOR_ROI_TOP_RATIO,
    MOTION_COOLDOWN_SECONDS,
    MOTION_SCORE_THRESHOLD,
    MOTION_SETTLE_SECONDS,
    PERIODIC_EVAL_SECONDS,
    PERIODIC_RECHECK_SECONDS,
    TEMPORAL_VOTING_MIN_HITS,
    TEMPORAL_VOTING_WINDOW,
    USE_LETTERBOX_PREPROCESSING,
)
from core.inference import (
    draw_detection_overlay,
    letterbox_to_square,
    load_model,
    predict_ensemble,
    predict_single,
)
from core.motion import DetectorState, MotionSettlementWatcher

__all__ = [
    "BASE_DIR",
    "CLASSES",
    "CONFIDENCE_THRESHOLD",
    "CONFIDENCE_UNCERTAIN_LABEL",
    "DEFAULT_AUDIO_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_STREAMLIT_HOST",
    "DEFAULT_STREAMLIT_PORT",
    "DetectorState",
    "ENSEMBLE_ENABLED",
    "FLOOR_ROI_TOP_RATIO",
    "MOTION_COOLDOWN_SECONDS",
    "MOTION_SCORE_THRESHOLD",
    "MOTION_SETTLE_SECONDS",
    "MotionSettlementWatcher",
    "PERIODIC_EVAL_SECONDS",
    "PERIODIC_RECHECK_SECONDS",
    "TEMPORAL_VOTING_MIN_HITS",
    "TEMPORAL_VOTING_WINDOW",
    "USE_LETTERBOX_PREPROCESSING",
    "draw_detection_overlay",
    "letterbox_to_square",
    "load_model",
    "predict_ensemble",
    "predict_single",
]
