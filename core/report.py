"""
Modul utilitas untuk memuat dan mengurai laporan evaluasi dan artefak training.
Menjadi satu-satunya sumber kebenaran (Single Source of Truth) untuk angka performa
agar tidak ada angka metrik atau KPI yang diketik manual / hardcode di kode atau UI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import (
    EVALUATION_REPORT_PATH,
    REPORTS_DIR,
    TRAINING_REPORTS_DIR,
)


def load_evaluation_report(report_path: Path | str | None = None) -> Optional[Dict[str, Any]]:
    """Memuat data laporan evaluasi dari file JSON.

    Returns:
        Dict berisi data laporan jika file valid, atau None jika belum dievaluasi.
    """
    path = Path(report_path) if report_path else EVALUATION_REPORT_PATH
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def get_evaluation_metrics_summary(report_path: Path | str | None = None) -> Dict[str, Any]:
    """Mengambil ringkasan metrik dari laporan evaluasi tanpa angka fallback hardcoded.

    Returns:
        Dict berisi ringkasan metrik (atau status 'belum dievaluasi' jika file tidak ada).
    """
    data = load_evaluation_report(report_path)
    if not data or "comparison" not in data:
        return {
            "status": "belum dievaluasi",
            "evaluated": False,
            "single": None,
            "ensemble": None,
            "ensemble_threshold": None,
        }

    comp = data.get("comparison", {})
    return {
        "status": "terukur",
        "evaluated": True,
        "split": data.get("dataset_split", "test"),
        "total_samples": data.get("total_samples", 0),
        "confidence_threshold": data.get("confidence_threshold", 0.70),
        "timestamp": data.get("evaluation_timestamp", "-"),
        "single": comp.get("single"),
        "ensemble": comp.get("ensemble"),
        "ensemble_threshold": comp.get("ensemble_threshold"),
    }


def get_training_artifacts_dir() -> Path:
    """Mengembalikan direktori penyimpanan artefak training yang terlacak (tracked)."""
    return TRAINING_REPORTS_DIR
