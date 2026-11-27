"""
Skrip audit integritas dataset:
1. Menghitung jumlah citra per kelas dan per split.
2. Mendeteksi duplikat identik (MD5) dan near-duplicate (Perceptual Hash / pHash).
3. Mendeteksi dan melaporkan kebocoran data (data leakage) antar split (train vs val/test).
4. Tidak menghapus file apa pun secara otomatis; hanya menghasilkan laporan analisis.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from PIL import Image
import imagehash


def compute_hashes(image_path: Path) -> tuple[str, imagehash.ImageHash]:
    """Menghitung MD5 checksum dan Perceptual Hash (pHash) untuk satu berkas citra."""
    with open(image_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    with Image.open(image_path) as img:
        phash = imagehash.phash(img)

    return md5, phash


def audit_dataset(
    dataset_dir: str | Path,
    max_hamming_distance: int = 4,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Melakukan audit menyeluruh terhadap struktur dan integritas dataset."""
    root = Path(dataset_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {root}")

    print("=" * 70)
    print(f"  AUDIT DATASET: {root.name}")
    print(f"  Lokasi: {root}")
    print("=" * 70)

    # 1. Pindai seluruh split dan kelas
    splits = [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    distribution: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_images: list[dict[str, Any]] = []

    print("\n[1] Memindai berkas citra...")
    for split in sorted(splits):
        split_dir = root / split
        classes = [c.name for c in split_dir.iterdir() if c.is_dir()]
        for cls_name in sorted(classes):
            cls_dir = split_dir / cls_name
            files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            distribution[split][cls_name] = len(files)

            for f in files:
                try:
                    md5, phash = compute_hashes(f)
                    all_images.append({
                        "path": str(f),
                        "rel_path": str(f.relative_to(root)),
                        "filename": f.name,
                        "split": split,
                        "class": cls_name,
                        "md5": md5,
                        "phash": str(phash),
                        "phash_obj": phash,
                    })
                except Exception as e:
                    print(f"  [PERINGATAN] Gagal membaca {f.name}: {e}")

    total_images = len(all_images)
    print(f"Total citra berhasil dianalisis: {total_images}")

    # Cetak tabel distribusi
    print("\n" + "=" * 55)
    print("  DISTRIBUSI CITRA PER SPLIT DAN KELAS")
    print("=" * 55)
    all_classes = sorted({item["class"] for item in all_images})
    header = f"{'Split':<12} | " + " | ".join(f"{c:<10}" for c in all_classes) + " | Total"
    print(header)
    print("-" * len(header))
    for split in sorted(splits):
        row_counts = [distribution[split][c] for c in all_classes]
        row_str = f"{split:<12} | " + " | ".join(f"{cnt:<10}" for cnt in row_counts) + f" | {sum(row_counts)}"
        print(row_str)
    print("=" * 55)

    # 2. Deteksi Duplikat Identik (MD5)
    print("\n[2] Memeriksa duplikat identik (MD5)...")
    md5_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for img in all_images:
        md5_map[img["md5"]].append(img)

    exact_duplicates: list[dict[str, Any]] = []
    cross_split_exact_leaks: list[dict[str, Any]] = []

    for md5_val, items in md5_map.items():
        if len(items) > 1:
            splits_involved = sorted({item["split"] for item in items})
            entry = {
                "md5": md5_val,
                "count": len(items),
                "splits": splits_involved,
                "files": [it["rel_path"] for it in items],
            }
            exact_duplicates.append(entry)
            if len(splits_involved) > 1:
                cross_split_exact_leaks.append(entry)

    print(f"  Ditemukan {len(exact_duplicates)} kelompok duplikat identik.")
    if cross_split_exact_leaks:
        print(f"  🚨 PERINGATAN KEBOCORAN EKSAK: {len(cross_split_exact_leaks)} kelompok citra muncul di split berbeda!")
        for leak in cross_split_exact_leaks[:5]:
            print(f"    - MD5 {leak['md5'][:8]} di split {leak['splits']}: {leak['files']}")
    else:
        print("  ✓ Tidak ada kebocoran duplikat eksak antar split.")

    # 3. Deteksi Near-Duplicate / Perceptual Hash Leakage Antar Split
    print(f"\n[3] Memeriksa near-duplicate antar split (Hamming Distance <= {max_hamming_distance})...")
    near_duplicates_cross_split: list[dict[str, Any]] = []

    # Kelompokkan citra berdasarkan split untuk perbandingan lintas split
    split_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_images:
        split_groups[item["split"]].append(item)

    split_names = sorted(split_groups.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            list1 = split_groups[s1]
            list2 = split_groups[s2]

            for item1 in list1:
                h1 = item1["phash_obj"]
                for item2 in list2:
                    h2 = item2["phash_obj"]
                    dist = h1 - h2
                    if dist <= max_hamming_distance:
                        near_duplicates_cross_split.append({
                            "split_a": s1,
                            "file_a": item1["rel_path"],
                            "class_a": item1["class"],
                            "split_b": s2,
                            "file_b": item2["rel_path"],
                            "class_b": item2["class"],
                            "hamming_distance": int(dist),
                        })

    print(f"  Ditemukan {len(near_duplicates_cross_split)} pasangan near-duplicate lintas split.")
    if near_duplicates_cross_split:
        print(f"  🚨 PERINGATAN KEBOCORAN DATA: Citra sangat mirip ditemukan di split berbeda!")
        for nd in near_duplicates_cross_split[:10]:
            print(f"    - Jarak {nd['hamming_distance']}: [{nd['split_a']}] {nd['file_a']} <---> [{nd['split_b']}] {nd['file_b']}")

    # 4. Ringkasan & Ekspor Hasil
    report = {
        "dataset_name": root.name,
        "dataset_path": str(root),
        "total_images": total_images,
        "distribution": {s: dict(cls_dict) for s, cls_dict in distribution.items()},
        "exact_duplicates_count": len(exact_duplicates),
        "cross_split_exact_leaks_count": len(cross_split_exact_leaks),
        "cross_split_exact_leaks": cross_split_exact_leaks,
        "near_duplicates_cross_split_count": len(near_duplicates_cross_split),
        "near_duplicates_cross_split": near_duplicates_cross_split,
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] Laporan audit berhasil disimpan ke: {out_p}")

    print("\n" + "=" * 70)
    print("  KESIMPULAN AUDIT DATASET:")
    if cross_split_exact_leaks or near_duplicates_cross_split:
        print("  ⚠️ DITEMUKAN INDIKASI KEBOCORAN DATA (DATA LEAKAGE) ANTAR SPLIT.")
        print("  Disarankan untuk mereorganisasi dataset menggunakan Group Split.")
    else:
        print("  ✓ DATASET BERSIH DARI KEBOCORAN LINTAS SPLIT.")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Integritas Dataset dan Deteksi Kebocoran")
    parser.add_argument("--data", default="dataset", help="Direktori dataset")
    parser.add_argument("--hamming-dist", type=int, default=4, help="Jarak Hamming maksimum untuk near-duplicate (default 4)")
    parser.add_argument("--output", default="reports/dataset_audit.json", help="Path keluaran file laporan JSON")
    args = parser.parse_args()

    audit_dataset(dataset_dir=args.data, max_hamming_distance=args.hamming_dist, output_json=args.output)
