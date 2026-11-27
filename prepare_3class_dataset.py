import os
import shutil
import zipfile
import random
from PIL import Image, ImageEnhance

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TEMP_DIR = os.path.join(BASE_DIR, "temp_kosong")
ZIP_PATH = os.path.join(os.path.expanduser("~"), "Downloads", "Kosong.zip")

def extract_and_prepare():
    print("=" * 60)
    print("  MENYIAPKAN KELAS BARU 'KOSONG' KE DATASET (3-CLASS)")
    print("=" * 60)

    # 1. Pastikan direktori tujuan bersih
    for split in ['train', 'val']:
        target_dir = os.path.join(DATASET_DIR, split, "KOSONG")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

    # 2. Ekstrak zip ke folder temporer
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    valid_images = []
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                extracted_path = z.extract(name, TEMP_DIR)
                try:
                    with Image.open(extracted_path) as img:
                        img.verify()
                    valid_images.append(extracted_path)
                except Exception as e:
                    print(f"File rusak diabaikan: {name} ({e})")

    print(f"Total gambar valid diekstrak dari Kosong.zip: {len(valid_images)}")
    random.shuffle(valid_images)

    # 3. Stratified Split 80% Train : 20% Val
    # 22 gambar -> 17 train (77.3%), 5 val (22.7%)
    val_count = max(4, int(len(valid_images) * 0.2))
    train_count = len(valid_images) - val_count
    
    train_raw = valid_images[:train_count]
    val_raw = valid_images[train_count:]

    print(f"Pembagian dasar: {len(train_raw)} Train, {len(val_raw)} Val")

    # 4. Simpan ke val/KOSONG tanpa modifikasi/augmentasi
    val_dir = os.path.join(DATASET_DIR, "val", "KOSONG")
    for i, path in enumerate(val_raw):
        dst = os.path.join(val_dir, f"val_kosong_{i+1:02d}.jpg")
        with Image.open(path) as img:
            img.convert("RGB").save(dst, "JPEG", quality=95)
    print(f"-> Disimpan ke val/KOSONG: {len(val_raw)} gambar asli")

    # 5. Simpan dan seimbangkan train/KOSONG dengan augmentasi offline
    # Agar rasio seimbang dengan BERANTAKAN (~133) dan RAPI (~116)
    train_dir = os.path.join(DATASET_DIR, "train", "KOSONG")
    generated_count = 0

    for i, path in enumerate(train_raw):
        with Image.open(path) as raw_img:
            img_rgb = raw_img.convert("RGB")
            
            # 5a. Simpan gambar asli
            base_name = f"train_kosong_{i+1:02d}_orig.jpg"
            img_rgb.save(os.path.join(train_dir, base_name), "JPEG", quality=95)
            generated_count += 1
            
            # 5b. Horizontal Flip
            hf_img = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
            hf_img.save(os.path.join(train_dir, f"train_kosong_{i+1:02d}_hf.jpg"), "JPEG", quality=95)
            generated_count += 1
            
            # 5c. Brightness Variations (Agak redup & agak terang)
            enhancer = ImageEnhance.Brightness(img_rgb)
            dim_img = enhancer.enhance(0.85)
            dim_img.save(os.path.join(train_dir, f"train_kosong_{i+1:02d}_dim.jpg"), "JPEG", quality=95)
            bright_img = enhancer.enhance(1.15)
            bright_img.save(os.path.join(train_dir, f"train_kosong_{i+1:02d}_bright.jpg"), "JPEG", quality=95)
            generated_count += 2
            
            # 5d. Slight Rotation (-8 deg dan +8 deg)
            rot_cw = img_rgb.rotate(-8, resample=Image.BICUBIC, expand=False)
            rot_cw.save(os.path.join(train_dir, f"train_kosong_{i+1:02d}_rot_cw.jpg"), "JPEG", quality=95)
            rot_ccw = img_rgb.rotate(8, resample=Image.BICUBIC, expand=False)
            rot_ccw.save(os.path.join(train_dir, f"train_kosong_{i+1:02d}_rot_ccw.jpg"), "JPEG", quality=95)
            generated_count += 2

    print(f"-> Disimpan ke train/KOSONG: {generated_count} gambar (original + balanced variations)")

    # 6. Hapus cache lama ultralytics
    for cache_f in ['train.cache', 'val.cache']:
        cp = os.path.join(DATASET_DIR, cache_f)
        if os.path.exists(cp):
            os.remove(cp)

    # 7. Bersihkan folder temporer
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    # 8. Tampilkan rekap dataset akhir
    print("\n" + "=" * 60)
    print("  REKAP DISTRIBUSI DATASET 3 KELAS")
    print("=" * 60)
    for split in ['train', 'val']:
        print(f"[{split.upper()}]")
        split_path = os.path.join(DATASET_DIR, split)
        total_split = 0
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                cnt = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_split += cnt
                print(f"  - {cls:<12}: {cnt} gambar")
        print(f"  Total {split}: {total_split} gambar\n")

if __name__ == "__main__":
    extract_and_prepare()
