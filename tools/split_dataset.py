"""
Skrip pembagian dataset berbasis grup (Group Split) menjadi 3 bagian: train / val / test.
Mencegah kebocoran data (data leakage) akibat foto burst atau sesi pemotretan yang sama
tersebar ke split berbeda.

Fitur:
1. Menentukan grup secara otomatis dari metadata nama file (Kelompok_X, NIM 24-XXX, origin ID Kosong).
2. Mendukung pemuatan metadata grup kustom via berkas JSON (--group-metadata).
3. Membagi secara terstratifikasi per kelas ke 3 split: train, val, dan test.
4. Idempotent: membersihkan folder tujuan sebelum menyalin, tidak menumpuk data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any

from PIL import Image


def infer_group_id(filename: str) -> str:
    """Menyimpulkan identitas grup/sesi pengambilan dari nama file citra."""
    fn_lower = filename.lower()

    # 1. Foto kelas KOSONG
    if "kosong" in fn_lower:
        # Ekstrak nomor seri gambar unik: misal train_kosong_01_orig.jpg -> KOSONG_01
        m_num = re.search(r'kosong_(\d+)', fn_lower)
        if m_num:
            return f"KOSONG_SERIES_{m_num.group(1)}"
        return "KOSONG_GENERAL"

    # 2. Format baru dengan kode NIM/mahasiswa: misal new_berantakan_020_24-034 Berantakan 3.jpg
    m_new = re.search(r'(\d{2}-\d{3})', filename)
    if m_new:
        return f"SESSION_{m_new.group(1)}"

    # 3. Format lama dengan nama kelompok: misal old_berantakan_041_Kelompok_5_...
    m_old = re.search(r'(kelompok_\d+)', fn_lower)
    if m_old:
        return f"OLD_{m_old.group(1).upper()}"

    # 4. Fallback jika ada pola tanggal IMG_YYYYMMDD
    m_date = re.search(r'(img_\d{8})', fn_lower)
    if m_date:
        return f"DATE_{m_date.group(1).upper()}"

    # Fallback terakhir menggunakan nama dasar tanpa ekstensi
    return f"SINGLE_{Path(filename).stem}"


def split_dataset_by_group(
    source_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    custom_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Membagi seluruh dataset menjadi train / val / test per grup."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Total rasio train + val + test harus 1.0"
    random.seed(seed)

    src_root = Path(source_dir).resolve()
    dst_root = Path(output_dir).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {src_root}")

    print("=" * 70)
    print("  PEMBAGIAN DATASET BERBASIS GRUP (GROUP SPLIT: TRAIN / VAL / TEST)")
    print(f"  Sumber  : {src_root}")
    print(f"  Tujuan  : {dst_root}")
    print(f"  Rasio   : {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {test_ratio*100:.0f}% Test (Seed: {seed})")
    print("=" * 70)

def build_leak_free_groups(
    src_root: Path,
    valid_exts: set[str],
    custom_meta: dict[str, str],
) -> dict[str, dict[str, list[Path]]]:
    """Mengelompokkan berkas citra ke dalam grup bebas kebocoran.

    Jika dua sesi/mahasiswa memiliki citra duplikat (MD5 identik), kedua sesi
    digabungkan ke dalam satu klaster grup yang sama (connected component)
    sehingga dijamin masuk ke split yang sama.
    """
    import hashlib

    # 1. Kumpulkan semua file dan hitung MD5
    class_files: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)

    for root, _, files in os.walk(src_root):
        rel = Path(root).relative_to(src_root)
        parts = rel.parts
        if not parts:
            continue
        cls_name = parts[-1].upper()
        if cls_name not in ("BERANTAKAN", "RAPI", "KOSONG"):
            continue

        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in valid_exts:
                grp_id = custom_meta.get(f, infer_group_id(f))
                try:
                    with open(p, "rb") as fp:
                        md5 = hashlib.md5(fp.read()).hexdigest()
                    class_files[cls_name].append((p, grp_id, md5))
                except Exception:
                    pass

    # 2. Gabungkan grup yang berbagi citra identik (Disjoint Set Union / DSU)
    class_groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for cls_name, file_list in class_files.items():
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            if parent.setdefault(x, x) != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Hubungkan grup yang berbagi MD5 identik
        md5_groups: dict[str, set[str]] = defaultdict(set)
        for _, grp_id, md5 in file_list:
            md5_groups[md5].add(grp_id)

        for md5, grps in md5_groups.items():
            grps_list = list(grps)
            for i in range(1, len(grps_list)):
                union(grps_list[0], grps_list[i])

        # Bentuk grup final yang terklaster bebas kebocoran
        for p, grp_id, _ in file_list:
            cluster_id = find(grp_id)
            class_groups[cls_name][cluster_id].append(p)

    return class_groups


def split_dataset_by_group(
    source_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    custom_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Membagi seluruh dataset menjadi train / val / test per grup."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Total rasio train + val + test harus 1.0"
    random.seed(seed)

    src_root = Path(source_dir).resolve()
    dst_root = Path(output_dir).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Folder sumber tidak ditemukan: {src_root}")

    print("=" * 70)
    print("  PEMBAGIAN DATASET BERBASIS GRUP (GROUP SPLIT: TRAIN / VAL / TEST)")
    print(f"  Sumber  : {src_root}")
    print(f"  Tujuan  : {dst_root}")
    print(f"  Rasio   : {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {test_ratio*100:.0f}% Test (Seed: {seed})")
    print("=" * 70)

    # Muat metadata grup kustom bila tersedia
    custom_meta: dict[str, str] = {}
    if custom_metadata_path and Path(custom_metadata_path).exists():
        with open(custom_metadata_path, "r", encoding="utf-8") as f:
            custom_meta = json.load(f)
        print(f"Memuat {len(custom_meta)} metadata grup kustom dari: {custom_metadata_path}")

    # 1. Kumpulkan grup bebas kebocoran
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    class_groups = build_leak_free_groups(src_root, valid_exts, custom_meta)

    # 2. Bersihkan dan siapkan struktur folder tujuan
    splits = ["train", "val", "test"]
    for split in splits:
        for cls_name in ["BERANTAKAN", "RAPI", "KOSONG"]:
            target_p = dst_root / split / cls_name
            if target_p.exists():
                shutil.rmtree(target_p)
            target_p.mkdir(parents=True, exist_ok=True)

    # 3. Lakukan Stratified Group Split
    summary: dict[str, dict[str, int]] = {s: defaultdict(int) for s in splits}
    assigned_groups: dict[str, str] = {}

    for cls_name, groups_dict in class_groups.items():
        total_cls_imgs = sum(len(imgs) for imgs in groups_dict.values())
        targets = {
            "train": total_cls_imgs * train_ratio,
            "val": total_cls_imgs * val_ratio,
            "test": total_cls_imgs * test_ratio,
        }
        alloc_counts: dict[str, int] = {s: 0 for s in splits}
        alloc_groups: dict[str, list[str]] = {s: [] for s in splits}

        # Urutkan grup secara descending agar grup besar dialokasikan terlebih dahulu,
        # dengan pengacakan terkontrol pada grup berukuran sama
        sorted_groups = sorted(
            groups_dict.keys(),
            key=lambda g: (len(groups_dict[g]), random.random()),
            reverse=True,
        )

        for grp in sorted_groups:
            n_imgs = len(groups_dict[grp])
            # Pilih split yang memiliki defisit alokasi relatif terbesar
            best_split = max(
                splits,
                key=lambda s: (targets[s] - alloc_counts[s]) / max(targets[s], 1.0),
            )
            alloc_groups[best_split].append(grp)
            alloc_counts[best_split] += n_imgs

        # Salin file ke masing-masing split
        for split_name in splits:
            for grp in alloc_groups[split_name]:
                assigned_groups[f"{cls_name}:{grp}"] = split_name
                for src_file in groups_dict[grp]:
                    dest_file = dst_root / split_name / cls_name / src_file.name
                    shutil.copy2(src_file, dest_file)
                    summary[split_name][cls_name] += 1

    # Hapus file cache ultralytics lama di folder tujuan
    for cache_f in ["train.cache", "val.cache", "test.cache"]:
        cp = dst_root / cache_f
        if cp.exists():
            cp.unlink()

    print("\n" + "=" * 65)
    print("  HASIL DISTRIBUSI GROUP SPLIT 3-BAGIAN")
    print("=" * 65)
    header = f"{'Split':<10} | {'BERANTAKAN':<12} | {'RAPI':<12} | {'KOSONG':<12} | Total Citra"
    print(header)
    print("-" * len(header))
    for s in splits:
        c_ber = summary[s]["BERANTAKAN"]
        c_rap = summary[s]["RAPI"]
        c_kos = summary[s]["KOSONG"]
        tot = c_ber + c_rap + c_kos
        print(f"{s:<10} | {c_ber:<12} | {c_rap:<12} | {c_kos:<12} | {tot}")
    print("=" * 65)
    print("Catatan: Test set bersifat independen dan TIDAK boleh dipakai untuk early stopping.")

    report = {
        "source": str(src_root),
        "destination": str(dst_root),
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "seed": seed,
        "distribution": {s: dict(cls_dict) for s, cls_dict in summary.items()},
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Dataset Berbasis Grup (Train/Val/Test)")
    parser.add_argument("--source", default="dataset", help="Direktori dataset sumber")
    parser.add_argument("--output", default="dataset", help="Direktori dataset tujuan")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Rasio train (default 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Rasio val (default 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Rasio test (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--group-metadata", default=None, help="Path file JSON metadata grup kustom jika ada")
    args = parser.parse_args()

    split_dataset_by_group(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        custom_metadata_path=args.group_metadata,
    )
