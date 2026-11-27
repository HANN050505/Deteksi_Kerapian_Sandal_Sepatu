"""
Modul utilitas untuk menggabungkan citra dari folder eksternal ke dalam dataset berformat YOLOv8-cls.
Mendukung argumen CLI fleksibel untuk direktori sumber citra baru dan direktori dataset tujuan.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple
from PIL import Image

from core.config import BASE_DIR, DATASET_DIR


def is_valid_image(filepath: str | Path) -> bool:
    """Memverifikasi integritas file citra."""
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    filepath_str = str(filepath).lower()
    if not filepath_str.endswith(valid_exts):
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_images_from_dir(folder: str | Path) -> List[Path]:
    """Mengumpulkan seluruh file citra valid dari sebuah folder (rekursif)."""
    p = Path(folder)
    if not p.exists():
        return []
    valid_files: List[Path] = []
    for root, _, filenames in os.walk(p):
        for f in filenames:
            file_path = Path(root) / f
            if is_valid_image(file_path):
                valid_files.append(file_path)
    return sorted(valid_files)


def merge_and_split(
    new_berantakan_dir: str | Path | None = None,
    new_rapi_dir: str | Path | None = None,
    target_dataset_dir: str | Path = DATASET_DIR,
    backup_existing: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """Menggabungkan citra baru dan melakukan stratified split ke direktori target."""
    random.seed(seed)
    target_path = Path(target_dataset_dir).resolve()
    backup_path = target_path.parent / f"{target_path.name}_backup_merge"

    print("=" * 65)
    print("  UTILITAS PENGGABUNGAN DATASET BERANTAKAN & RAPI")
    print("=" * 65)
    print(f"Direktori Target Dataset: {target_path}")

    new_dirs = {
        "BERANTAKAN": Path(new_berantakan_dir).resolve() if new_berantakan_dir else None,
        "RAPI": Path(new_rapi_dir).resolve() if new_rapi_dir else None,
    }

    # Validasi direktori sumber
    for cls_name, p in new_dirs.items():
        if p is None or not p.exists():
            print(f"[CATATAN] Folder sumber untuk kelas '{cls_name}' tidak ditentukan atau tidak ada: {p}")
            print(f"Gunakan argumen --berantakan dan --rapi untuk menentukan folder sumber citra baru.")
            return

    # Backup dataset lama bila ada
    if backup_existing and target_path.exists():
        if not backup_path.exists():
            print(f"Membuat cadangan dataset ke: {backup_path} ...")
            shutil.copytree(target_path, backup_path)
            print("Cadangan berhasil dibuat.")

    # Kumpulkan citra baru
    stats: Dict[str, Dict[str, int]] = {}
    for cls_name, folder in new_dirs.items():
        files = collect_images_from_dir(folder)
        print(f"[{cls_name}] Ditemukan {len(files)} citra valid dari: {folder}")
        if not files:
            print(f"[PERINGATAN] Tidak ada citra yang ditemukan untuk kelas {cls_name}.")
            continue

        shuffled = list(files)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]

        train_dst = target_path / "train" / cls_name
        val_dst = target_path / "val" / cls_name
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(train_files):
            dst = train_dst / f"new_{cls_name.lower()}_{i:04d}_{src.name}"
            shutil.copy2(src, dst)

        for i, src in enumerate(val_files):
            dst = val_dst / f"new_{cls_name.lower()}_{i:04d}_{src.name}"
            shutil.copy2(src, dst)

        stats[cls_name] = {
            "total": len(shuffled),
            "train": len(train_files),
            "val": len(val_files),
        }

    print("\n" + "=" * 65)
    print("  RINGKASAN HASIL PENGGABUNGAN DATASET")
    print("=" * 65)
    for cls_name, s in stats.items():
        print(f" - {cls_name:12s}: Total {s['total']:3d} -> Train: {s['train']:3d}, Val: {s['val']:2d}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Penggabungan citra baru ke dataset YOLOv8-cls")
    parser.add_argument("--berantakan", type=str, default=None, help="Folder sumber citra BERANTAKAN")
    parser.add_argument("--rapi", type=str, default=None, help="Folder sumber citra RAPI")
    parser.add_argument("--target", type=str, default=str(DATASET_DIR), help="Folder target dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proporsi data training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Seed generator acak")
    parser.add_argument("--no-backup", action="store_true", help="Jangan buat folder backup")

    args = parser.parse_args()
    merge_and_split(
        new_berantakan_dir=args.berantakan,
        new_rapi_dir=args.rapi,
        target_dataset_dir=args.target,
        backup_existing=not args.no_backup,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
