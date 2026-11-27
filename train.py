"""
Skrip pelatihan (training) model klasifikasi YOLOv8n-cls (3 Kelas: BERANTAKAN, RAPI, KOSONG).
Telah diverifikasi sesuai dokumentasi & kode internal Ultralytics classify_augmentations:
- Menghapus parameter augmentasi deteksi objek yang tidak berlaku (degrees, shear, perspective).
- Mengaktifkan augmentasi klasifikasi yang valid (RandAugment, ColorJitter, RandomErasing, RandomResizedCrop, Flip).
- Menyediakan argumen CLI fleksibel untuk seed, device (cpu/cuda), batch, dan epochs.
- Menghilangkan klaim sepihak 'Target KPI tercapai: YA' pada set validasi.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

from ultralytics import YOLO

from core.config import BASE_DIR, DATASET_DIR, DEFAULT_MODEL_PATH, IMG_SIZE, RUNS_DIR


def train_classifier(
    data_dir: str | Path = DATASET_DIR,
    epochs: int = 30,
    batch_size: int = 16,
    img_size: int = IMG_SIZE,
    device: str = "cpu",
    seed: int = 42,
    model_name: str = "yolov8_sandal_v3",
) -> None:
    """Melatih model YOLOv8n-cls dengan konfigurasi augmentasi klasifikasi terverifikasi."""
    data_path = Path(data_dir).resolve()
    project_dir = Path(RUNS_DIR).resolve()
    model_export_dst = Path(DEFAULT_MODEL_PATH).resolve()
    backup_old_model = model_export_dst.parent / "best_v2_backup.pt"

    # 1. Cadangkan bobot model lama bila ada dan belum dicadangkan
    if model_export_dst.exists() and not backup_old_model.exists():
        backup_old_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_export_dst, backup_old_model)
        print(f"[BACKUP] Bobot model lama dicadangkan ke: {backup_old_model}")

    print("=" * 70)
    print("  PELATIHAN YOLOV8 NANO CLASSIFIER (3 KELAS)")
    print("=" * 70)
    print(f"Direktori Dataset : {data_path}")
    print(f"Backbone Model    : yolov8n-cls.pt (Transfer Learning)")
    print(f"Epochs / Batch    : {epochs} Epochs / Batch Size {batch_size} / Img Size {img_size}")
    print(f"Device / Seed     : {device} (Seed: {seed})")
    print("Optimizer         : AdamW + Cosine Annealing LR (lr0=0.001, lrf=0.01)")
    print("Augmentasi Aktif  : RandAugment, Horizontal Flip (50%), Color Jitter (HSV),")
    print("                    Random Resized Crop (scale=0.2), Random Erasing (20%)")
    print("Catatan Parameter : Parameter 'degrees', 'shear', dan 'perspective' telah dihapus")
    print("                    karena terbukti tidak berlaku pada task klasifikasi Ultralytics.")
    print("=" * 70)

    model = YOLO("yolov8n-cls.pt")

    start_time = time.time()
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        workers=0,  # Multiprocessing 0 paling stabil di OS Windows
        device=device,
        seed=seed,
        project=str(project_dir),
        name=model_name,
        exist_ok=True,
        verbose=True,
        # Optimizer & Scheduler
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        label_smoothing=0.05,
        # Augmentasi Resmi Ultralytics classify_augmentations
        fliplr=0.5,
        scale=0.2,
        auto_augment="randaugment",
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        erasing=0.2,
        patience=12,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining selesai dalam {elapsed/60:.2f} menit!")

    # 2. Perbarui bobot terbaik ke direktori models/
    best_weights = project_dir / model_name / "weights" / "best.pt"
    if best_weights.exists():
        model_export_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, model_export_dst)
        print(f"[EXPORT] Bobot model terbaik diperbarui di: {model_export_dst}")
    else:
        print(f"[PERINGATAN] File bobot {best_weights} tidak ditemukan.")

    # 3. Evaluasi akhir pada set validasi (pelaporan metrik terukur objektif)
    print("\n" + "=" * 70)
    print("  EVALUASI MODEL PADA SET VALIDASI (HASIL TERUKUR)")
    print("=" * 70)
    eval_model = YOLO(str(model_export_dst))
    val_metrics = eval_model.val(data=str(data_path), device=device, workers=0)

    print("\n=== METRIK VALIDASI TERUKUR ===")
    print(f"Top-1 Accuracy : {val_metrics.top1 * 100:.2f}%")
    print(f"Top-5 Accuracy : {val_metrics.top5 * 100:.2f}%")
    print(f"Daftar Kelas   : {eval_model.names}")
    print("Catatan: Evaluasi resmi tanpa kebocoran harus diukur pada TEST SET independen via evaluate.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan YOLOv8n-cls 3-Kelas")
    parser.add_argument("--data", default=str(DATASET_DIR), help="Direktori dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Jumlah epochs training")
    parser.add_argument("--batch", type=int, default=16, help="Ukuran batch")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Resolusi gambar input")
    parser.add_argument("--device", default="cpu", help="Device pelatihan: 'cpu' atau 'cuda' / '0'")
    parser.add_argument("--seed", type=int, default=42, help="Seed generator acak")
    parser.add_argument("--name", default="yolov8_sandal_v3", help="Nama sub-folder eksperimen runs")
    args = parser.parse_args()

    train_classifier(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        seed=args.seed,
        model_name=args.name,
    )
