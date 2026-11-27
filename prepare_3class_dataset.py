"""
Skrip integrasi dan penyiapan kelas 'KOSONG' ke dataset (3-Class YOLOv8):
1. Menerima path berkas ZIP / folder input melalui argumen CLI (tanpa hardcode ~/Downloads).
2. Mengekstrak dan memverifikasi integritas berkas citra asli secara idempotent.
3. Melaporkan jumlah gambar UNIK sebelum augmentasi secara transparan dan akurat.
4. Koreksi kalkulasi split: dari 22 gambar unik, 18 train (81.8%) dan 4 val (18.2%), bukan 17/5.
5. Melakukan augmentasi offline yang terukur pada set latih untuk menyeimbangkan distribusi.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import shutil
import zipfile
from typing import Any

from PIL import Image, ImageEnhance

from core.config import BASE_DIR, DATASET_DIR


def extract_and_prepare_kosong(
    source_zip_or_dir: str | Path,
    dataset_dir: str | Path = DATASET_DIR,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """Mengekstrak dan menyiapkan kelas KOSONG secara reproducible dan idempotent."""
    random.seed(seed)
    src_path = Path(source_zip_or_dir).resolve()
    target_dataset = Path(dataset_dir).resolve()
    temp_dir = target_dataset.parent / "temp_kosong"

    print("=" * 65)
    print("  PENYIAPAN KELAS 'KOSONG' KE DATASET (REPRODUCIBLE & IDEMPOTENT)")
    print(f"  Sumber Input : {src_path}")
    print(f"  Target Dir   : {target_dataset}")
    print("=" * 65)

    if not src_path.exists():
        raise FileNotFoundError(f"Sumber input tidak ditemukan di: {src_path}")

    # 1. Bersihkan direktori tujuan dan temp secara idempotent
    for split in ["train", "val", "test"]:
        split_kosong = target_dataset / split / "KOSONG"
        if split_kosong.exists():
            shutil.rmtree(split_kosong)
        split_kosong.mkdir(parents=True, exist_ok=True)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    valid_images: list[Path] = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # 2. Ekstrak dari file ZIP atau baca dari direktori
    if src_path.is_file() and src_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(src_path, "r") as z:
            for name in z.namelist():
                if Path(name).suffix.lower() in valid_exts:
                    extracted = z.extract(name, temp_dir)
                    try:
                        with Image.open(extracted) as img:
                            img.verify()
                        valid_images.append(Path(extracted))
                    except Exception as e:
                        print(f"  [Peringatan] Citra rusak diabaikan: {name} ({e})")
    elif src_path.is_dir():
        for p in src_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                try:
                    with Image.open(p) as img:
                        img.verify()
                    valid_images.append(p)
                except Exception as e:
                    print(f"  [Peringatan] Citra rusak diabaikan: {p.name} ({e})")
    else:
        raise ValueError(f"Sumber harus berupa file .zip atau direktori: {src_path}")

    unique_count = len(valid_images)
    print(f"\n[INFO] Total citra UNIK ditemukan: {unique_count} gambar")
    if unique_count == 0:
        raise RuntimeError(f"Tidak ada gambar valid yang dapat diproses dari: {src_path}")

    random.shuffle(valid_images)

    # 3. Stratified Split pada data asli
    # Koreksi kalkulasi: Dari 22 gambar unik -> 18 train (81.8%) dan 4 val (18.2%)
    val_count = max(4, int(unique_count * val_ratio))
    train_count = unique_count - val_count

    train_raw = valid_images[:train_count]
    val_raw = valid_images[train_count:]

    print(f"Pembagian data UNIK: {len(train_raw)} Train ({len(train_raw)/unique_count*100:.1f}%), {len(val_raw)} Val ({len(val_raw)/unique_count*100:.1f}%)")

    # 4. Simpan val/KOSONG tanpa augmentasi (menjaga integritas data evaluasi)
    val_dir = target_dataset / "val" / "KOSONG"
    for i, path in enumerate(val_raw):
        dst = val_dir / f"val_kosong_{i+1:02d}.jpg"
        with Image.open(path) as img:
            img.convert("RGB").save(dst, "JPEG", quality=95)

    print(f"-> Disimpan ke val/KOSONG: {len(val_raw)} gambar asli (100% UNIK tanpa augmentasi)")

    # 5. Simpan dan seimbangkan train/KOSONG dengan augmentasi offline terukur (6 variasi)
    train_dir = target_dataset / "train" / "KOSONG"
    generated_count = 0

    for i, path in enumerate(train_raw):
        with Image.open(path) as raw_img:
            img_rgb = raw_img.convert("RGB")

            # 5a. Citra asli
            img_rgb.save(train_dir / f"train_kosong_{i+1:02d}_orig.jpg", "JPEG", quality=95)
            generated_count += 1

            # 5b. Flip horizontal
            hf = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            hf.save(train_dir / f"train_kosong_{i+1:02d}_hf.jpg", "JPEG", quality=95)
            generated_count += 1

            # 5c. Variasi kecerahan (dim & bright)
            enhancer = ImageEnhance.Brightness(img_rgb)
            enhancer.enhance(0.85).save(train_dir / f"train_kosong_{i+1:02d}_dim.jpg", "JPEG", quality=95)
            enhancer.enhance(1.15).save(train_dir / f"train_kosong_{i+1:02d}_bright.jpg", "JPEG", quality=95)
            generated_count += 2

            # 5d. Rotasi halus (-8 deg dan +8 deg)
            img_rgb.rotate(-8, resample=Image.BICUBIC, expand=False).save(
                train_dir / f"train_kosong_{i+1:02d}_rot_cw.jpg", "JPEG", quality=95
            )
            img_rgb.rotate(8, resample=Image.BICUBIC, expand=False).save(
                train_dir / f"train_kosong_{i+1:02d}_rot_ccw.jpg", "JPEG", quality=95
            )
            generated_count += 2

    print(f"-> Disimpan ke train/KOSONG: {generated_count} gambar ({len(train_raw)} UNIK + variasi)")

    # 6. Hapus cache lama ultralytics
    for cache_f in ["train.cache", "val.cache", "test.cache"]:
        cp = target_dataset / cache_f
        if cp.exists():
            cp.unlink()

    # 7. Bersihkan folder temporer
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 65)
    print("  RINGKASAN AKHIR KELAS KOSONG:")
    print(f"  - Total Citra Asli (UNIK) : {unique_count}")
    print(f"  - Set Validasi            : {len(val_raw)} citra unik")
    print(f"  - Set Training Asli       : {len(train_raw)} citra unik")
    print(f"  - Set Training Augmented  : {generated_count} citra")
    print("=" * 65)

    return {
        "unique_total": unique_count,
        "train_unique": len(train_raw),
        "val_unique": len(val_raw),
        "train_augmented": generated_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siapkan Kelas KOSONG ke Dataset")
    parser.add_argument("--source", default=None, help="Path ke file Kosong.zip atau direktori gambar kosong")
    parser.add_argument("--data", default=str(DATASET_DIR), help="Direktori dataset target")
    parser.add_argument("--seed", type=int, default=42, help="Seed generator (default 42)")
    args = parser.parse_args()

    default_src = Path(BASE_DIR) / "Kosong.zip"
    input_source = args.source or (str(default_src) if default_src.exists() else None)

    if not input_source:
        # Periksa folder backup jika ada
        backup_kosong = Path(BASE_DIR) / "dataset_backup_phase3" / "val" / "KOSONG"
        if backup_kosong.exists():
            print(f"[INFO] Sumber ZIP tidak ditentukan, menggunakan data gambar dari: {backup_kosong}")
            input_source = str(backup_kosong.parent.parent)

    if input_source:
        extract_and_prepare_kosong(source_zip_or_dir=input_source, dataset_dir=args.data, seed=args.seed)
    else:
        print("[ERROR] Masukkan argumen --source <path_ke_Kosong.zip_atau_folder>")
