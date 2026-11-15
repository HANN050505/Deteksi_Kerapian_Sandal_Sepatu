import os
import shutil
import random
from PIL import Image

random.seed(42)

DATASET_DIR = r"D:\deteksi_sandal\dataset"
BACKUP_DIR = r"D:\deteksi_sandal\dataset_backup_v1"

NEW_DIRS = {
    "BERANTAKAN": r"C:\Users\Raihan aryanova\Downloads\berantakan-20261110T065157Z-1-001\berantakan",
    "RAPI": r"C:\Users\Raihan aryanova\Downloads\rapi-20261110T065157Z-1-001\rapi"
}

def is_valid_image(filepath):
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    if not filepath.lower().endswith(valid_exts):
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def collect_existing(cls_name):
    files = []
    for split in ['train', 'val']:
        folder = os.path.join(BACKUP_DIR, split, cls_name)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                full_path = os.path.join(folder, f)
                if os.path.isfile(full_path) and is_valid_image(full_path):
                    files.append(full_path)
    return files

def collect_new(cls_name):
    folder = NEW_DIRS[cls_name]
    files = []
    if os.path.exists(folder):
        for root, _, filenames in os.walk(folder):
            for f in filenames:
                full_path = os.path.join(root, f)
                if os.path.isfile(full_path) and is_valid_image(full_path):
                    files.append(full_path)
    return files

def run_merge_and_split():
    print("=" * 60)
    print("  PENGGABUNGAN DATASET LAMA & BARU + STRATIFIED SPLIT")
    print("=" * 60)

    # 1. Backup existing dataset if backup doesn't exist
    if not os.path.exists(BACKUP_DIR):
        print(f"Membuat cadangan dataset lama ke: {BACKUP_DIR} ...")
        shutil.copytree(DATASET_DIR, BACKUP_DIR)
        print("Cadangan berhasil dibuat!")
    else:
        print(f"Cadangan dataset lama sudah ada di: {BACKUP_DIR}")

    # 2. Collect all images per class
    all_data = {}
    for cls in ["BERANTAKAN", "RAPI"]:
        old_imgs = collect_existing(cls)
        new_imgs = collect_new(cls)
        print(f"[{cls}] Gambar Lama: {len(old_imgs)} | Gambar Baru: {len(new_imgs)} | Total: {len(old_imgs) + len(new_imgs)}")
        
        items = []
        for i, p in enumerate(old_imgs):
            items.append((p, f"old_{cls.lower()}_{i:03d}_{os.path.basename(p)}"))
        for i, p in enumerate(new_imgs):
            items.append((p, f"new_{cls.lower()}_{i:03d}_{os.path.basename(p)}"))
            
        random.shuffle(items)
        all_data[cls] = items

    # 3. Clean target train/val directories
    for split in ['train', 'val']:
        for cls in ["BERANTAKAN", "RAPI"]:
            target_folder = os.path.join(DATASET_DIR, split, cls)
            if os.path.exists(target_folder):
                shutil.rmtree(target_folder)
            os.makedirs(target_folder, exist_ok=True)

    # Clean old cache files
    for cache_f in ['train.cache', 'val.cache']:
        cp = os.path.join(DATASET_DIR, cache_f)
        if os.path.exists(cp):
            os.remove(cp)

    # 4. Copy with 80% train / 20% val split
    stats = {}
    for cls, items in all_data.items():
        total = len(items)
        split_idx = int(total * 0.8)
        train_items = items[:split_idx]
        val_items = items[split_idx:]
        
        train_dir = os.path.join(DATASET_DIR, 'train', cls)
        val_dir = os.path.join(DATASET_DIR, 'val', cls)
        
        for src, name in train_items:
            shutil.copy2(src, os.path.join(train_dir, name))
            
        for src, name in val_items:
            shutil.copy2(src, os.path.join(val_dir, name))
            
        stats[cls] = {
            "total": total,
            "train": len(train_items),
            "val": len(val_items)
        }

    print("\n" + "=" * 60)
    print("  DISTRIBUSI DATASET FINAL HASIL PENGGABUNGAN")
    print("=" * 60)
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_all = sum(s['total'] for s in stats.values())
    
    for cls, s in stats.items():
        pct = (s['total'] / total_all) * 100
        print(f" - {cls:12s}: Total {s['total']:3d} ({pct:.1f}%) -> Train: {s['train']:3d}, Val: {s['val']:2d}")
    print(f"Total Keseluruhan : {total_all} gambar (Train: {total_train}, Val: {total_val})")
    print("=" * 60)

if __name__ == "__main__":
    run_merge_and_split()
