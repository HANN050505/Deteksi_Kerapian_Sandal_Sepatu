import os
import sys
import argparse
import time
import threading
import cv2
from PIL import Image
import torch
from ultralytics import YOLO

# Batasi thread CPU PyTorch agar ringan
torch.set_num_threads(2)

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

def play_audio(audio_path):
    def _play():
        try:
            if os.path.exists(audio_path):
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error audio: {e}")
    t = threading.Thread(target=_play, daemon=True)
    t.start()

def pick_image_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Pilih Foto Sandal / Sepatu untuk Diuji",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"), ("All Files", "*.*")]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"Error dialog file: {e}")
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "best.pt")
DEFAULT_AUDIO = os.path.join(BASE_DIR, "audio", "peringatan.mp3")

def test_image(image_path, model_path=None, audio_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL
    if audio_path is None:
        audio_path = DEFAULT_AUDIO
    if not os.path.exists(image_path):
        print(f"[ERROR] File gambar tidak ditemukan: {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] File model tidak ditemukan: {model_path}")
        return

    print(f"\n[INFO] Memuat model dari: {model_path} ...")
    model = YOLO(model_path)
    
    print(f"[INFO] Menguji gambar dengan Batch Spatial Ensemble: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Gagal membaca gambar: {image_path}")
        return

    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 1. Global View
    crop_top = int(h * 0.35)
    floor_crop = img_pil.crop((0, crop_top, w, h))
    flip_img = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    # Batch forward pass (hemat waktu 60%)
    batch_results = model.predict(source=[img_pil, floor_crop, flip_img], verbose=False, device='cpu', imgsz=224)
    names = batch_results[0].names

    p_global = {names[i].upper(): float(batch_results[0].probs.data[i]) for i in range(len(names))}
    p_crop = {names[i].upper(): float(batch_results[1].probs.data[i]) for i in range(len(names))}
    p_flip = {names[i].upper(): float(batch_results[2].probs.data[i]) for i in range(len(names))}

    # Soft Voting Ensemble
    classes = ["BERANTAKAN", "RAPI", "KOSONG"]
    p_ens = {}
    for c in classes:
        p_ens[c] = (
            0.50 * p_global.get(c, 0.0) +
            0.35 * p_crop.get(c, 0.0) +
            0.15 * p_flip.get(c, 0.0)
        )
    tot = sum(p_ens.values()) or 1.0
    p_ens = {c: val / tot for c, val in p_ens.items()}
    final_label = max(p_ens, key=p_ens.get)
    confidence = p_ens[final_label]

    print("\n" + "="*55)
    print(f" HASIL KEPUTUSAN ENSEMBLE AI : {final_label} ({confidence*100:.2f}%)")
    print("="*55)
    print(" Detail Konsensus Ensemble:")
    for c in classes:
        print(f" - {c:12s}: {p_ens[c]*100:.2f}% (Global: {p_global[c]*100:.1f}%, Crop: {p_crop[c]*100:.1f}%, Flip: {p_flip[c]*100:.1f}%)")
    print("="*55)

    annotated_frame = frame.copy()

    # Gambar batas Drop Zone / Zona Lantai Pemantauan yang rapi
    roi_y = int(h * 0.35)
    cv2.rectangle(annotated_frame, (8, roi_y), (w - 8, h - 8), (255, 190, 0), 2)
    cv2.putText(
        annotated_frame,
        "ZONA PEMANTAUAN LANTAI / RAK SANDAL",
        (15, roi_y - 10 if roi_y > 25 else roi_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 190, 0),
        2,
        cv2.LINE_AA
    )

    if final_label == "BERANTAKAN":
        banner_color = (0, 0, 220) # Red
        status_text = f"PERINGATAN! BERANTAKAN ({confidence*100:.1f}%)"
        sub_text = "Sandal/sepatu berserakan di depan pintu masuk lab!"
        print("\n[AUDIO] Memutar peringatan suara...")
        play_audio(audio_path)
    elif final_label == "RAPI":
        banner_color = (0, 200, 0) # Green
        status_text = f"STATUS: RAPI ({confidence*100:.1f}%)"
        sub_text = "Sandal/sepatu tertata rapi di rak / teratur berdampingan"
    else: # KOSONG
        banner_color = (220, 140, 20) # Blue-cyan (BGR)
        status_text = f"STATUS: KOSONG / STANDBY ({confidence*100:.1f}%)"
        sub_text = "Area lantai bersih (tidak ada sandal atau sepatu)"

    # Overlay banner on annotated_frame
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        annotated_frame = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        h, w, _ = annotated_frame.shape

    overlay = annotated_frame.copy()
    banner_height = max(75, int(h * 0.12))
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

    font_scale_main = max(0.65, w / 900.0)
    font_scale_sub = max(0.5, w / 1200.0)

    cv2.putText(annotated_frame, status_text, (15, int(banner_height * 0.45)), cv2.FONT_HERSHEY_DUPLEX, font_scale_main, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, sub_text, (15, int(banner_height * 0.82)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255, 255, 255), 1, cv2.LINE_AA)

    win_title = f"Uji Foto Sandal (Ensemble) - {os.path.basename(image_path)}"
    cv2.imshow(win_title, annotated_frame)
    print("\nTekan sembarang tombol pada jendela gambar untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uji Foto Model Deteksi Sandal")
    parser.add_argument("--image", default=None, help="Path ke file foto sandal/sepatu")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path bobot model")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Path audio file")
    args = parser.parse_args()

    target_image = args.image
    if not target_image:
        print("\nMemilih gambar melalui jendela file dialog...")
        target_image = pick_image_gui()

    if target_image and os.path.exists(target_image):
        test_image(target_image, model_path=args.model, audio_path=args.audio)
    else:
        print("Pengujian dibatalkan: Tidak ada file gambar yang dipilih.")
