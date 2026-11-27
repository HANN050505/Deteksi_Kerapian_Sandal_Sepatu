"""
Skrip evaluasi performa model YOLOv8n-cls (3 Kelas: BERANTAKAN, RAPI, KOSONG):
1. Menghitung akurasi, precision, recall, dan F1-score per kelas.
2. Menghitung interval kepercayaan 95% (Wilson Score Interval) untuk akurasi.
3. Menghitung False Alarm Rate (FAR) khusus: proporsi RAPI yang salah diprediksi sebagai BERANTAKAN.
4. Membandingkan 3 skenario secara empiris pada data uji yang sama:
   - Model Tunggal (Single Perspective)
   - Spatial Ensemble (Global + Floor Crop + Flip TTA)
   - Spatial Ensemble + Confidence Threshold (Status RAGU)
5. Menyimpan laporan komparasi ke format JSON dan visualisasi Confusion Matrix ke PNG di reports/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core import (
    CLASSES,
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_UNCERTAIN_LABEL,
    DEFAULT_MODEL_PATH,
    draw_detection_overlay,
    load_model,
    predict_ensemble,
    predict_single,
)


def compute_wilson_interval(successes: int, total: int, z: float = 1.95996) -> tuple[float, float]:
    """Menghitung Wilson Score Interval 95% untuk proporsi keberhasilan (akurasi).

    Args:
        successes: Jumlah prediksi benar.
        total: Total jumlah sampel evaluasi.
        z: Nilai z-score untuk tingkat kepercayaan 95% (1.96).

    Returns:
        Tuple (lower_bound, upper_bound) dalam rentang 0.0 - 1.0.
    """
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z**2) / (4.0 * (total**2))))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def evaluate_split(
    model: Any,
    dataset_dir: str | Path,
    split: str = "test",
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Menjalankan evaluasi komparatif komprehensif pada satu split data."""
    root = Path(dataset_dir).resolve()
    split_dir = root / split

    if not split_dir.exists():
        fallback_split = "val" if split == "test" else "test"
        if (root / fallback_split).exists():
            print(f"[PERINGATAN] Split '{split}' tidak ditemukan di {root}. Menggunakan fallback '{fallback_split}'...")
            split = fallback_split
            split_dir = root / split
        else:
            raise FileNotFoundError(f"Split direktori tidak ditemukan: {split_dir}")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    dataset_items: list[tuple[Path, str]] = []

    for cls_name in CLASSES:
        cls_folder = split_dir / cls_name
        if cls_folder.exists():
            for f in sorted(cls_folder.iterdir()):
                if f.is_file() and f.suffix.lower() in valid_exts:
                    dataset_items.append((f, cls_name))

    total_samples = len(dataset_items)
    print("=" * 70)
    print(f"  EVALUASI PERFORMA MODEL PADA SET: '{split.upper()}'")
    print(f"  Lokasi: {split_dir} | Total Sampel: {total_samples}")
    print("=" * 70)

    if total_samples == 0:
        raise ValueError(f"Tidak ada sampel citra pada {split_dir}")

    # Koleksi hasil prediksi untuk 3 konfigurasi
    modes = ["single", "ensemble", "ensemble_threshold"]
    predictions: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}

    print("\n[1] Menjalankan inferensi pada seluruh sampel uji...")
    t_start = time.time()
    for idx, (img_path, true_label) in enumerate(dataset_items):
        # Jalankan satu batch forward pass ensemble (sudah mencakup global, crop, flip)
        p_ens = predict_ensemble(model, img_path, enable_ensemble=True, confidence_threshold=0.0)

        # 1. Single Model perspective (diambil langsung dari evaluasi global view)
        p_glob = p_ens.get("details", {}).get("global", {})
        top_single = max(p_glob, key=p_glob.get) if p_glob else p_ens["raw_label"]
        conf_single = p_glob.get(top_single, p_ens["confidence"])

        # 2. Ensemble perspective
        top_ens = p_ens["raw_label"]
        conf_ens = p_ens["confidence"]

        # 3. Ensemble + Confidence Threshold
        is_conf = bool(conf_ens >= conf_threshold)
        top_thresh = top_ens if is_conf else CONFIDENCE_UNCERTAIN_LABEL

        predictions["single"].append({
            "pred_label": top_single,
            "conf": conf_single,
            "true_label": true_label,
            "path": str(img_path),
        })
        predictions["ensemble"].append({
            "pred_label": top_ens,
            "conf": conf_ens,
            "true_label": true_label,
            "path": str(img_path),
        })
        predictions["ensemble_threshold"].append({
            "pred_label": top_thresh,
            "raw_pred": top_ens,
            "conf": conf_ens,
            "true_label": true_label,
            "path": str(img_path),
        })

    elapsed = time.time() - t_start
    print(f"Inferensi selesai dalam {elapsed:.2f} detik ({(elapsed/total_samples)*1000:.1f} ms/sampel)")

    # 2. Hitung Metrik Evaluasi untuk Setiap Mode
    metrics_summary: dict[str, Any] = {}

    for mode in modes:
        preds = predictions[mode]
        correct = sum(1 for p in preds if p["pred_label"] == p["true_label"])
        acc = correct / total_samples
        ci_low, ci_high = compute_wilson_interval(correct, total_samples)

        # Hitung False Alarm Rate (FAR): RAPI diprediksi BERANTAKAN
        true_rapi = [p for p in preds if p["true_label"] == "RAPI"]
        false_alarms = [p for p in true_rapi if p["pred_label"] == "BERANTAKAN"]
        far = (len(false_alarms) / len(true_rapi)) if true_rapi else 0.0

        # Confusion Matrix
        eval_classes = list(CLASSES) + ([CONFIDENCE_UNCERTAIN_LABEL] if mode == "ensemble_threshold" else [])
        cm = {c_true: {c_pred: 0 for c_pred in eval_classes} for c_true in CLASSES}
        for p in preds:
            cm[p["true_label"]][p["pred_label"]] += 1

        # Precision, Recall, F1 per kelas
        class_metrics: dict[str, dict[str, float]] = {}
        for c in CLASSES:
            tp = cm[c][c]
            fp = sum(cm[other][c] for other in CLASSES if other != c)
            fn = sum(cm[c][pred_c] for pred_c in eval_classes if pred_c != c)

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            class_metrics[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": sum(cm[c].values()),
            }

        macro_f1 = float(np.mean([m["f1_score"] for m in class_metrics.values()]))

        metrics_summary[mode] = {
            "accuracy": round(acc, 4),
            "accuracy_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "correct_count": correct,
            "total_count": total_samples,
            "false_alarm_rate": round(far, 4),
            "false_alarm_count": len(false_alarms),
            "total_true_rapi": len(true_rapi),
            "macro_f1": round(macro_f1, 4),
            "class_metrics": class_metrics,
            "confusion_matrix": cm,
        }

    # Cetak Tabel Perbandingan
    print("\n" + "=" * 78)
    print("  HASIL KOMPARASI EMPIRIS (EVALUASI TEST SET)")
    print("=" * 78)
    print(f"{'Metrik':<30} | {'Single Model':<14} | {'Ensemble':<14} | {'Ensemble+Thresh':<14}")
    print("-" * 78)
    acc_s = f"{metrics_summary['single']['accuracy']*100:.2f}%"
    acc_e = f"{metrics_summary['ensemble']['accuracy']*100:.2f}%"
    acc_t = f"{metrics_summary['ensemble_threshold']['accuracy']*100:.2f}%"
    print(f"{'Top-1 Accuracy':<30} | {acc_s:<14} | {acc_e:<14} | {acc_t:<14}")

    ci_s = f"[{metrics_summary['single']['accuracy_ci_95'][0]*100:.1f}%, {metrics_summary['single']['accuracy_ci_95'][1]*100:.1f}%]"
    ci_e = f"[{metrics_summary['ensemble']['accuracy_ci_95'][0]*100:.1f}%, {metrics_summary['ensemble']['accuracy_ci_95'][1]*100:.1f}%]"
    ci_t = f"[{metrics_summary['ensemble_threshold']['accuracy_ci_95'][0]*100:.1f}%, {metrics_summary['ensemble_threshold']['accuracy_ci_95'][1]*100:.1f}%]"
    print(f"{'Wilson 95% Conf. Interval':<30} | {ci_s:<14} | {ci_e:<14} | {ci_t:<14}")

    far_s = f"{metrics_summary['single']['false_alarm_rate']*100:.2f}% ({metrics_summary['single']['false_alarm_count']}/{metrics_summary['single']['total_true_rapi']})"
    far_e = f"{metrics_summary['ensemble']['false_alarm_rate']*100:.2f}% ({metrics_summary['ensemble']['false_alarm_count']}/{metrics_summary['ensemble']['total_true_rapi']})"
    far_t = f"{metrics_summary['ensemble_threshold']['false_alarm_rate']*100:.2f}% ({metrics_summary['ensemble_threshold']['false_alarm_count']}/{metrics_summary['ensemble_threshold']['total_true_rapi']})"
    print(f"{'False Alarm Rate (RAPI->BER)':<30} | {far_s:<14} | {far_e:<14} | {far_t:<14}")

    f1_s = f"{metrics_summary['single']['macro_f1']:.4f}"
    f1_e = f"{metrics_summary['ensemble']['macro_f1']:.4f}"
    f1_t = f"{metrics_summary['ensemble_threshold']['macro_f1']:.4f}"
    print(f"{'Macro F1-Score':<30} | {f1_s:<14} | {f1_e:<14} | {f1_t:<14}")
    print("=" * 78)

    # 3. Simpan Gambar Confusion Matrix
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_modes = [
        ("single", "Single Model", axes[0]),
        ("ensemble", "Spatial Ensemble (0.50/0.35/0.15)", axes[1]),
        ("ensemble_threshold", f"Ensemble + Threshold ({conf_threshold*100:.0f}%)", axes[2]),
    ]

    for key, title, ax in plot_modes:
        cm_dict = metrics_summary[key]["confusion_matrix"]
        labels_y = list(CLASSES)
        labels_x = list(cm_dict[labels_y[0]].keys())
        matrix_arr = np.array([[cm_dict[r][c] for c in labels_x] for r in labels_y])

        sns.heatmap(matrix_arr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels_x, yticklabels=labels_y)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel("True Class", fontsize=10)
        ax.set_xlabel("Predicted Class", fontsize=10)

    plt.tight_layout()
    cm_plot_path = out_dir / "confusion_matrix_eval.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"\n[INFO] Confusion matrix visual disimpan ke: {cm_plot_path}")

    # 4. Simpan Laporan JSON Lengkap
    final_report = {
        "dataset_split": split,
        "dataset_path": str(root),
        "total_samples": total_samples,
        "confidence_threshold": conf_threshold,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparison": metrics_summary,
    }
    report_json_path = out_dir / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"[INFO] Laporan evaluasi JSON disimpan ke: {report_json_path}")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Performa Model YOLOv8 3-Kelas")
    parser.add_argument("--data", default="dataset", help="Direktori dataset")
    parser.add_argument("--split", default="test", help="Nama split yang diuji (default: test, fallback ke val)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path model .pt")
    parser.add_argument("--conf-threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Ambang batas keyakinan")
    parser.add_argument("--reports-dir", default="reports", help="Direktori penyimpanan laporan")
    args = parser.parse_args()

    yolo_model = load_model(args.model)
    evaluate_split(
        model=yolo_model,
        dataset_dir=args.data,
        split=args.split,
        conf_threshold=args.conf_threshold,
        reports_dir=args.reports_dir,
    )
