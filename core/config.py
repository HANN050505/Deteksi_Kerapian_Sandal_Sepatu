"""
Konfigurasi terpusat untuk sistem Smart Shoe Monitor.
Menjadi satu-satunya sumber kebenaran (single source of truth) untuk seluruh parameter:
ROI, threshold deteksi, timing gerakan, bobot ensemble, dan path default.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# -----------------------------------------------------------------------------
# Path Direktori Proyek
# -----------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH: Final[Path] = BASE_DIR / "models" / "best.pt"
DEFAULT_AUDIO_PATH: Final[Path] = BASE_DIR / "audio" / "peringatan.mp3"
DATASET_DIR: Final[Path] = BASE_DIR / "dataset"
RUNS_DIR: Final[Path] = BASE_DIR / "runs"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"
TRAINING_REPORTS_DIR: Final[Path] = REPORTS_DIR / "training"
EVALUATION_REPORT_PATH: Final[Path] = REPORTS_DIR / "evaluation_report.json"

# Nama run model aktif yang sedang dideploy
ACTIVE_RUN_NAME: Final[str] = "yolov8_sandal_group_split"

# -----------------------------------------------------------------------------
# Kelas Model & Dimensi Citra
# -----------------------------------------------------------------------------
CLASSES: Final[tuple[str, ...]] = ("BERANTAKAN", "RAPI", "KOSONG")
IMG_SIZE: Final[int] = 224

# -----------------------------------------------------------------------------
# Parameter Spasial ROI (Region-of-Interest) Lantai
# -----------------------------------------------------------------------------
# Rasio batas atas area lantai (0.35 = 35% dari atas diabaikan, 65% area bawah dipantau).
# Konstanta tunggal ini digunakan secara seragam oleh watcher, crop inferensi, dan visual overlay.
FLOOR_ROI_TOP_RATIO: Final[float] = 0.35

# -----------------------------------------------------------------------------
# Parameter Deteksi Gerakan (Motion Settlement Watcher)
# -----------------------------------------------------------------------------
MOTION_DOWNSCALE_SIZE: Final[tuple[int, int]] = (240, 160)
MOTION_DIFF_THRESHOLD: Final[int] = 28
MOTION_SCORE_THRESHOLD: Final[float] = 0.025
MOTION_SETTLE_SECONDS: Final[float] = 1.8
MOTION_COOLDOWN_SECONDS: Final[float] = 10.0

# -----------------------------------------------------------------------------
# Parameter Inferensi & Ensemble
# -----------------------------------------------------------------------------
# Saklar untuk mengaktifkan / menonaktifkan Test-Time Spatial Ensemble (default: False / OFF)
# Evaluasi empiris membuktikan single model setara (95.59%) dengan latensi ~10x lebih hemat.
ENSEMBLE_ENABLED: Final[bool] = False

# Bobot voting probabilistik ensemble (global, floor crop, horizontal flip)
ENSEMBLE_WEIGHT_GLOBAL: Final[float] = 0.50
ENSEMBLE_WEIGHT_CROP: Final[float] = 0.35
ENSEMBLE_WEIGHT_FLIP: Final[float] = 0.15

# Ambang keyakinan minimum (confidence threshold)
CONFIDENCE_THRESHOLD: Final[float] = 0.70
CONFIDENCE_UNCERTAIN_LABEL: Final[str] = "RAGU"

# -----------------------------------------------------------------------------
# Parameter Temporal Voting & Evaluasi Real-Time (Fase 2)
# -----------------------------------------------------------------------------
# Temporal voting: minimal K dari N evaluasi berurutan harus 'BERANTAKAN' sebelum audio berbunyi
TEMPORAL_VOTING_WINDOW: Final[int] = 3
TEMPORAL_VOTING_MIN_HITS: Final[int] = 2

# Evaluasi periodik saat STANDBY (mendeteksi kondisi berantakan sejak awal)
PERIODIC_EVAL_SECONDS: Final[float] = 15.0

# Re-check berkala setelah audio berbunyi / dalam masa pemantauan
PERIODIC_RECHECK_SECONDS: Final[float] = 10.0

# Preprocessing Letterbox/Square Pad (menjaga seluruh bidang pandang tanpa terpotong center crop)
USE_LETTERBOX_PREPROCESSING: Final[bool] = True

# -----------------------------------------------------------------------------
# Pengaturan Server Streamlit
# -----------------------------------------------------------------------------
DEFAULT_STREAMLIT_HOST: Final[str] = "127.0.0.1"
DEFAULT_STREAMLIT_PORT: Final[int] = 8501
