import os
import shutil
import random

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
SOURCE_DIRS = {
    'BERANTAKAN': os.path.join(DOWNLOADS_DIR, "BERANTAKAN-20261108T182328Z-1-001", "BERANTAKAN"),
    'RAPI': os.path.join(DOWNLOADS_DIR, "RAPI-20261108T182328Z-1-001", "RAPI")
}

DEST_DIR = os.path.join(BASE_DIR, "dataset")

def collect_files(source_dir):
    collected = []
    for root, dirs, files in os.walk(source_dir):
        rel = os.path.relpath(root, source_dir)
        group_prefix = rel.replace('\\', '_').replace('/', '_').replace(' ', '_')
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, f)
                new_name = f"{group_prefix}_{f}" if group_prefix != '.' else f
                collected.append((src_path, new_name))
    return collected

def prepare():
    stats = {}
    for cls_name, src_dir in SOURCE_DIRS.items():
        files = collect_files(src_dir)
        random.shuffle(files)
        
        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        stats[cls_name] = {'total': len(files), 'train': len(train_files), 'val': len(val_files)}
        
        train_dst = os.path.join(DEST_DIR, 'train', cls_name)
        val_dst = os.path.join(DEST_DIR, 'val', cls_name)
        
        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(val_dst, exist_ok=True)
        
        for src, name in train_files:
            shutil.copy2(src, os.path.join(train_dst, name))
            
        for src, name in val_files:
            shutil.copy2(src, os.path.join(val_dst, name))

    print("=== Dataset Distribution ===")
    for cls_name, st in stats.items():
        print(f"[{cls_name}] Total: {st['total']} -> Train: {st['train']}, Val: {st['val']}")
    print("Dataset prepared successfully at:", DEST_DIR)

if __name__ == '__main__':
    prepare()
