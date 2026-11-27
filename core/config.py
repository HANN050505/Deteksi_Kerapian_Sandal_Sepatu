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
# Saklar untuk mengaktifkan / menonaktifkan Test-Time Spatial Ensemble
ENSEMBLE_ENABLED: Final[bool] = True

# Bobot voting probabilistik ensemble (global, floor crop, horizontal flip)
ENSEMBLE_WEIGHT_GLOBAL: Final[float] = 0.50
ENSEMBLE_WEIGHT_CROP: Final[float] = 0.35
ENSEMBLE_WEIGHT_FLIP: Final[float] = 0.15

# Ambang keyakinan minimum (confidence threshold)
CONFIDENCE_THRESHOLD: Final[float] = 0.70

# -----------------------------------------------------------------------------
# Pengaturan Server Streamlit
# -----------------------------------------------------------------------------
DEFAULT_STREAMLIT_HOST: Final[str] = "127.0.0.1"
DEFAULT_STREAMLIT_PORT: Final[int] = 8501
