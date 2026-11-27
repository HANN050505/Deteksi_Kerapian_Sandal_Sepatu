"""
Skrip uji mandiri (self-diagnostic) prototype Smart Shoe Monitor:
1. Memverifikasi kelengkapan lingkungan Python, modul core, dan dependensi.
2. Memverifikasi bobot model YOLOv8 (models/best.pt) dan kesesuaian 3 kelas.
3. Memverifikasi modul audio peringatan dan driver mixer.
4. Memverifikasi struktur dataset (train / val / test).
5. Benchmark latensi nyata menggunakan pipeline inferensi ensemble lengkap pada citra riil dataset.
6. Memeriksa ketersediaan kamera fisik dengan status jujur (kamera tidak ada = WARN, bukan PASS).
7. Sanity test inferensi: menguji 1 citra acak per kelas dari test set.
8. Menampilkan scorecard jujur dengan status PASS / WARN / FAIL terpisah tanpa klaim palsu.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

from core import (
    CLASSES,
    DEFAULT_AUDIO_PATH,
    DEFAULT_MODEL_PATH,
    DATASET_DIR,
    load_model,
    predict_ensemble,
)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


def run_diagnostics() -> dict[str, str]:
    print("=" * 70)
    print("     SMART SHOE MONITOR - DIAGNOSTIK MANDIRI PROTOTYPE (JUJUR)")
    print("=" * 70)

    results: dict[str, str] = {}

    # -------------------------------------------------------------------------
    # 1. CEK PYTHON & DEPENDENSI
    # -------------------------------------------------------------------------
    print("\n[1/7] Memeriksa Lingkungan Python & Dependensi...")
    modules = [
        ("ultralytics", "YOLOv8 Engine"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("streamlit", "Streamlit Web"),
        ("pygame", "Pygame Audio"),
        ("PIL", "Pillow Imaging"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas Dataframe"),
        ("matplotlib", "Matplotlib"),
        ("imagehash", "ImageHash (Perceptual Hash)"),
    ]
    missing_mods = []
    for mod_name, label in modules:
        try:
            __import__(mod_name)
            print(f"  [PASS] {label} ({mod_name}) terpasang.")
        except ImportError:
            print(f"  [FAIL] {label} ({mod_name}) BELUM terpasang.")
            missing_mods.append(mod_name)

    if not missing_mods:
        results["dependensi"] = "PASS"
        print("  -> Status Dependensi: PASS (Lengkap)")
    else:
        results["dependensi"] = "FAIL"
        print(f"  -> Status Dependensi: FAIL (Kurang: {', '.join(missing_mods)})")

    # -------------------------------------------------------------------------
    # 2. CEK FILE BOBOT MODEL & KELAS
    # -------------------------------------------------------------------------
    print("\n[2/7] Memeriksa Bobot Model YOLOv8 (models/best.pt)...")
    model_obj = None
    if DEFAULT_MODEL_PATH.exists():
        size_mb = DEFAULT_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"  [INFO] File model ditemukan: {DEFAULT_MODEL_PATH} ({size_mb:.2f} MB)")
        try:
            model_obj = load_model(DEFAULT_MODEL_PATH)
            model_classes = {name.upper() for name in model_obj.names.values()}
            expected_classes = set(CLASSES)

            if expected_classes.issubset(model_classes):
                print(f"  [PASS] Kelas Model Sesuai (3 Kelas): {sorted(model_classes)}")
                results["model"] = "PASS"
            else:
                print(f"  [FAIL] Kelas model tidak cocok. Ditemukan: {model_classes}, Diharapkan: {expected_classes}")
                results["model"] = "FAIL"
        except Exception as e:
            print(f"  [FAIL] Gagal memuat model: {e}")
            results["model"] = "FAIL"
    else:
        print(f"  [FAIL] File model tidak ditemukan di: {DEFAULT_MODEL_PATH}")
        results["model"] = "FAIL"

    # -------------------------------------------------------------------------
    # 3. CEK MODUL AUDIO PERINGATAN
    # -------------------------------------------------------------------------
    print("\n[3/7] Memeriksa Modul Audio Peringatan...")
    if DEFAULT_AUDIO_PATH.exists():
        size_kb = DEFAULT_AUDIO_PATH.stat().st_size / 1024
        print(f"  [INFO] File audio ditemukan: {DEFAULT_AUDIO_PATH} ({size_kb:.1f} KB)")
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            results["audio"] = "PASS"
            print("  [PASS] Sub-sistem driver audio mixer berhasil diinisialisasi.")
        except Exception as e:
            results["audio"] = "WARN"
            print(f"  [WARN] Driver audio sistem tidak tersedia: {e}")
    else:
        results["audio"] = "WARN"
        print(f"  [WARN] File audio tidak ditemukan di: {DEFAULT_AUDIO_PATH} (Audio peringatan tidak akan berbunyi)")

    # -------------------------------------------------------------------------
    # 4. CEK STRUKTUR DATASET (TRAIN / VAL / TEST)
    # -------------------------------------------------------------------------
    print("\n[4/7] Memeriksa Struktur Direktori Dataset...")
    dataset_splits = ["train", "val", "test"]
    split_counts: dict[str, dict[str, int]] = {}
    dataset_has_all = True
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for split in dataset_splits:
        split_p = Path(DATASET_DIR) / split
        split_counts[split] = {}
        if split_p.exists():
            for c in CLASSES:
                c_dir = split_p / c
                if c_dir.exists():
                    n = len([f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
                    split_counts[split][c] = n
                else:
                    split_counts[split][c] = 0
                    dataset_has_all = False
            total_s = sum(split_counts[split].values())
            print(f"  [INFO] Split '{split}': Total {total_s} citra {split_counts[split]}")
        else:
            print(f"  [WARN] Direktori split '{split}' belum ada di: {split_p}")
            dataset_has_all = False

    if dataset_has_all and all(sum(split_counts[s].values()) > 0 for s in dataset_splits):
        results["dataset"] = "PASS"
        print("  -> Status Dataset: PASS (Train / Val / Test lengkap)")
    elif any(sum(split_counts.get(s, {}).values()) > 0 for s in ["train", "val"]):
        results["dataset"] = "WARN"
        print("  -> Status Dataset: WARN (Hanya train/val parsial, test belum terpisah)")
    else:
        results["dataset"] = "FAIL"
        print("  -> Status Dataset: FAIL (Dataset kosong atau struktur rusak)")

    # -------------------------------------------------------------------------
    # 5. BENCHMARK LATENSI REALISTIS (PIPELINE NYATA PADA GAMBAR ASLI)
    # -------------------------------------------------------------------------
    print("\n[5/7] Menguji Benchmark Latensi Pipeline Nyata (Spatial Ensemble)...")
    if model_obj is not None:
        # Cari satu gambar riil dari dataset
        sample_img_path = None
        for split in ["test", "val", "train"]:
            candidate_dir = Path(DATASET_DIR) / split / "BERANTAKAN"
            if candidate_dir.exists():
                for f in candidate_dir.iterdir():
                    if f.suffix.lower() in valid_exts:
                        sample_img_path = f
                        break
            if sample_img_path:
                break

        if sample_img_path and sample_img_path.exists():
            print(f"  [INFO] Sampel citra nyata: {sample_img_path.name}")
            import cv2
            real_bgr = cv2.imread(str(sample_img_path))

            # Warmup pipeline (2 iterasi)
            for _ in range(2):
                _ = predict_ensemble(model_obj, real_bgr)

            # Pengukuran latensi nyata (5 iterasi pipeline lengkap)
            iters = 5
            t0 = time.time()
            for _ in range(iters):
                _ = predict_ensemble(model_obj, real_bgr)
            total_time = time.time() - t0
            avg_lat_ms = (total_time / iters) * 1000.0
            fps_est = 1000.0 / avg_lat_ms

            print(f"  [INFO] Rata-rata Latensi Nyata (Ensemble 3-View + Letterbox): {avg_lat_ms:.1f} ms (~{fps_est:.1f} eval/detik)")
            if avg_lat_ms < 1200:
                results["latensi"] = "PASS"
                print("  -> Status Performa: PASS (Sangat memadai untuk event-driven)")
            else:
                results["latensi"] = "WARN"
                print("  -> Status Performa: WARN (Komputasi CPU lambat)")
        else:
            results["latensi"] = "WARN"
            print("  [WARN] Tidak ada gambar sampel riil untuk benchmark.")
    else:
        results["latensi"] = "FAIL"
        print("  [FAIL] Benchmark dilewati karena model belum termuat.")

    # -------------------------------------------------------------------------
    # 6. CEK PERANGKAT KAMERA FISIK (JUJUR: KAMERA TIDAK ADA = WARN)
    # -------------------------------------------------------------------------
    print("\n[6/7] Memeriksa Ketersediaan Perangkat Kamera Fisik...")
    try:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                results["kamera"] = "PASS"
                print("  [PASS] Webcam perangkat keras (index 0) terdeteksi aktif dan menghasilkan frame.")
            else:
                results["kamera"] = "WARN"
                print("  [WARN] Kamera index 0 terbuka tetapi tidak dapat membaca frame (mungkin sedang dipakai aplikasi lain).")
        else:
            results["kamera"] = "WARN"
            print("  [WARN] Kamera fisik index 0 tidak terdeteksi. Sistem tetap dapat menggunakan IP Webcam HP via Wi-Fi atau fitur Upload Foto.")
    except Exception as e:
        results["kamera"] = "WARN"
        print(f"  [WARN] Error pengujian kamera: {e}")

    # -------------------------------------------------------------------------
    # 7. SANITY TEST: 1 CITRA PER KELAS DARI TEST SET
    # -------------------------------------------------------------------------
    print("\n[7/7] Menjalankan Sanity Test (1 Sampel per Kelas)...")
    if model_obj is not None:
        sanity_split = "test" if (Path(DATASET_DIR) / "test").exists() else "val"
        sanity_correct = 0
        sanity_total = 0

        for expected_cls in CLASSES:
            cls_folder = Path(DATASET_DIR) / sanity_split / expected_cls
            if cls_folder.exists():
                files = [f for f in cls_folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
                if files:
                    test_file = files[0]
                    sanity_total += 1
                    pred = predict_ensemble(model_obj, test_file, confidence_threshold=0.0)
                    predicted_cls = pred["raw_label"]
                    is_match = (predicted_cls == expected_cls)
                    if is_match:
                        sanity_correct += 1
                        print(f"  [PASS] Kelas '{expected_cls}': Prediksi '{predicted_cls}' ({pred['confidence']*100:.1f}%) -> COCOK")
                    else:
                        print(f"  [WARN] Kelas '{expected_cls}': Prediksi '{predicted_cls}' ({pred['confidence']*100:.1f}%) -> MISMATCH")

        if sanity_total == len(CLASSES) and sanity_correct == len(CLASSES):
            results["sanity_test"] = "PASS"
            print(f"  -> Sanity Test: PASS ({sanity_correct}/{sanity_total} kelas terverifikasi akurat)")
        elif sanity_correct > 0:
            results["sanity_test"] = "WARN"
            print(f"  -> Sanity Test: WARN ({sanity_correct}/{sanity_total} kelas cocok)")
        else:
            results["sanity_test"] = "FAIL"
            print(f"  -> Sanity Test: FAIL (Seluruh sampel sanity gagal diprediksi)")
    else:
        results["sanity_test"] = "FAIL"
        print("  [FAIL] Sanity test dilewati karena model gagal dimuat.")

    # -------------------------------------------------------------------------
    # REKAPITULASI SCORECARD JUJUR
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SCORECARD HASIL DIAGNOSTIK PROTOTYPE (OBJEKTIF)")
    print("=" * 70)
    pass_cnt = sum(1 for v in results.values() if v == "PASS")
    warn_cnt = sum(1 for v in results.values() if v == "WARN")
    fail_cnt = sum(1 for v in results.values() if v == "FAIL")

    for test_name, status in results.items():
        symbol = "✓ PASS" if status == "PASS" else ("⚠️ WARN" if status == "WARN" else "✗ FAIL")
        print(f"  - {test_name.replace('_', ' ').title():<25} : {symbol}")

    print("-" * 70)
    print(f"  TOTAL: PASS = {pass_cnt} | WARN = {warn_cnt} | FAIL = {fail_cnt}")

    if fail_cnt == 0:
        if warn_cnt == 0:
            print("  KESIMPULAN: Sistem prototype memenuhi seluruh kriteria verifikasi.")
        else:
            print("  KESIMPULAN: Sistem siap beroperasi dengan catatan/peringatan non-kritis di atas.")
    else:
        print("  KESIMPULAN: Sistem memiliki komponen FAIL yang harus diperbaiki sebelum demonstrasi.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_diagnostics()
