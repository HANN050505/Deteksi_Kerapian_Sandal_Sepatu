import os
import shutil
import time
from ultralytics import YOLO

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "dataset")
    project_dir = os.path.join(base_dir, "runs")
    model_name = "yolov8_sandal_v3"
    model_export_dst = os.path.join(base_dir, "models", "best.pt")
    backup_old_model = os.path.join(base_dir, "models", "best_v2_backup.pt")
    
    # 1. Backup old model if exists and not already backed up
    if os.path.exists(model_export_dst) and not os.path.exists(backup_old_model):
        shutil.copy2(model_export_dst, backup_old_model)
        print(f"Model lama dicadangkan ke: {backup_old_model}")

    print("=" * 65)
    print("  MEMULAI RETRAINING YOLOV8 CLASSIFIER 3 KELAS")
    print("  (BERANTAKAN, RAPI, KOSONG)")
    print("=" * 65)
    print("Dataset Directory :", data_dir)
    print("Backbone          : YOLOv8n-cls (Pretrained Transfer Learning)")
    print("Optimizer         : AdamW + Cosine LR Scheduler (lr0=0.001, lrf=0.01)")
    print("Augmentasi Aktif  : Flip-LR, Random Rotation (15°), Scale, Perspective,")
    print("                    HSV Jitter, Random Erasing (Cutout), Label Smoothing")
    print("=" * 65)

    model = YOLO("yolov8n-cls.pt")
    
    start_time = time.time()
    results = model.train(
        data=data_dir,
        epochs=30,
        imgsz=224,
        batch=16,
        workers=0,  # 0 paling stabil dan aman di Windows
        device="cpu",
        project=project_dir,
        name=model_name,
        exist_ok=True,
        verbose=True,
        # Optimizers & Scheduler
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        label_smoothing=0.05,
        # Data Augmentations
        fliplr=0.5,
        degrees=15.0,
        scale=0.2,
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        erasing=0.2,
        patience=12
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining selesai dalam {elapsed/60:.2f} menit!")

    # Copy best weights to models/best.pt
    best_weights = os.path.join(project_dir, model_name, "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy2(best_weights, model_export_dst)
        print(f"Bobot model terbaik diperbarui di: {model_export_dst}")
    else:
        print(f"Peringatan: File {best_weights} tidak ditemukan.")

    # Evaluasi akurasi akhir pada dataset validasi
    print("\n" + "=" * 65)
    print("  EVALUASI MODEL BARU PADA DATASET VALIDASI (3 KELAS)")
    print("=" * 65)
    eval_model = YOLO(model_export_dst)
    val_metrics = eval_model.val(data=data_dir, device="cpu", workers=0)
    
    print("\n=== HASIL METRIK VALIDASI ===")
    print(f"Top-1 Accuracy : {val_metrics.top1 * 100:.2f}%")
    print(f"Top-5 Accuracy : {val_metrics.top5 * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    train()
