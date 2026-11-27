"""
Script desktop pengujian citra foto sandal/sepatu menggunakan GUI file picker atau argumen CLI.
Menggunakan modul inferensi terpadu dari core.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time
from typing import Optional

import cv2
import pygame

from core import (
    CLASSES,
    DEFAULT_AUDIO_PATH,
    DEFAULT_MODEL_PATH,
    draw_detection_overlay,
    load_model,
    predict_ensemble,
)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


def play_audio(audio_path: str | Path) -> None:
    """Memutar berkas audio peringatan secara non-blocking di latar belakang."""
    def _play() -> None:
        try:
            if os.path.exists(audio_path):
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(str(audio_path))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error pemutaran audio: {e}")

    threading.Thread(target=_play, daemon=True).start()


def pick_image_gui() -> Optional[str]:
    """Membuka dialog file explorer sistem untuk memilih file citra."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Pilih Foto Sandal / Sepatu untuk Diuji",
            filetypes=[
                ("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        return file_path or None
    except Exception as e:
        print(f"Gagal membuka dialog file: {e}")
        return None


def test_image(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    audio_path: str | Path = DEFAULT_AUDIO_PATH,
    no_show: bool = False,
) -> None:
    """Menguji satu berkas citra, menampilkan overlay hasil, dan memicu audio bila berantakan."""
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] File gambar tidak ditemukan: {img_path}")
        return

    print(f"\n[INFO] Memuat model dari: {model_path} ...")
    model = load_model(model_path)

    print(f"[INFO] Menguji gambar dengan Batch Spatial Ensemble: {img_path.name}")
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[ERROR] Gagal membaca gambar: {img_path}")
        return

    # Evaluasi Ensemble melalui fungsi modul bersama
    pred = predict_ensemble(model, frame)
    final_label = pred["label"]
    confidence = pred["confidence"]
    probs = pred["probs"]
    details = pred.get("details", {})

    print("\n" + "=" * 55)
    print(f" HASIL KEPUTUSAN ENSEMBLE AI : {final_label} ({confidence*100:.2f}%)")
    print("=" * 55)
    print(" Detail Konsensus Ensemble:")
    p_global = details.get("global", {})
    p_crop = details.get("crop", {})
    p_flip = details.get("flip", {})
    for c in CLASSES:
        print(
            f" - {c:12s}: {probs.get(c, 0.0)*100:.2f}% "
            f"(Global: {p_global.get(c, 0.0)*100:.1f}%, "
            f"Crop: {p_crop.get(c, 0.0)*100:.1f}%, "
            f"Flip: {p_flip.get(c, 0.0)*100:.1f}%)"
        )
    print("=" * 55)

    if final_label == "BERANTAKAN":
        print("\n[AUDIO] Memutar peringatan suara...")
        play_audio(audio_path)

    annotated = draw_detection_overlay(frame, pred)

    if not no_show:
        # Batasi resolusi tampilan jendela bila citra terlalu besar
        h, w = annotated.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

        win_title = f"Uji Foto Sandal (Ensemble) - {img_path.name}"
        cv2.imshow(win_title, annotated)
        print("\nTekan sembarang tombol pada jendela gambar untuk menutup...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uji Foto Model Deteksi Sandal")
    parser.add_argument("--image", default=None, help="Path ke file foto sandal/sepatu")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path bobot model")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO_PATH), help="Path audio file")
    parser.add_argument("--no-show", action="store_true", help="Jangan tampilkan jendela GUI")
    args = parser.parse_args()

    target_image = args.image
    if not target_image:
        print("\nMemilih gambar melalui jendela file dialog...")
        target_image = pick_image_gui()

    if target_image and os.path.exists(target_image):
        test_image(target_image, model_path=args.model, audio_path=args.audio, no_show=args.no_show)
    else:
        print("Pengujian dibatalkan: Tidak ada file gambar yang dipilih.")
