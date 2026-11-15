import os
import sys
import argparse
import time
import threading
import cv2
from ultralytics import YOLO

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

def analyze_and_draw_shoe_pairs(image_bgr):
    if image_bgr is None:
        return None, 0, 0, 0
    annotated = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return annotated, 0, 0, 0
    scale = 640.0 / max(h, w)
    new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
    small = cv2.resize(image_bgr, (new_w, new_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    inv_scale = 1.0 / scale
    
    for c in contours:
        area = cv2.contourArea(c)
        if 800 < area < 45000:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = float(bh) / max(bw, 1)
            if 0.3 <= aspect <= 3.5:
                ox, oy = int(x * inv_scale), int(y * inv_scale)
                obw, obh = int(bw * inv_scale), int(bh * inv_scale)
                cx = ox + obw / 2.0
                cy = oy + obh / 2.0
                candidates.append({
                    'box': (ox, oy, obw, obh),
                    'center': (cx, cy),
                    'area': area,
                    'orig_w': obw,
                    'orig_h': obh
                })
                
    pairs = []
    unpaired = []
    matched = set()
    n = len(candidates)
    
    for i in range(n):
        if i in matched:
            continue
        best_j = -1
        min_dist = float('inf')
        ci = candidates[i]['center']
        ai = candidates[i]['area']
        max_allowed_dist = max(candidates[i]['orig_w'], candidates[i]['orig_h']) * 2.2
        
        for j in range(i+1, n):
            if j in matched:
                continue
            cj = candidates[j]['center']
            aj = candidates[j]['area']
            dist = np.hypot(ci[0] - cj[0], ci[1] - cj[1])
            area_ratio = min(ai, aj) / max(ai, aj)
            
            if dist < max_allowed_dist and area_ratio > 0.4:
                if dist < min_dist:
                    min_dist = dist
                    best_j = j
                    
        if best_j != -1:
            matched.add(i)
            matched.add(best_j)
            pairs.append((candidates[i], candidates[best_j]))
        else:
            matched.add(i)
            unpaired.append(candidates[i])
            
    for p1, p2 in pairs:
        b1, b2 = p1['box'], p2['box']
        cv2.rectangle(annotated, (b1[0], b1[1]), (b1[0]+b1[2], b1[1]+b1[3]), (0, 220, 0), 2)
        cv2.rectangle(annotated, (b2[0], b2[1]), (b2[0]+b2[2], b2[1]+b2[3]), (0, 220, 0), 2)
        c1 = (int(p1['center'][0]), int(p1['center'][1]))
        c2 = (int(p2['center'][0]), int(p2['center'][1]))
        cv2.line(annotated, c1, c2, (0, 255, 0), 2)
        min_x = min(b1[0], b2[0])
        min_y = min(b1[1], b2[1])
        cv2.putText(annotated, "PASANGAN RAPI", (min_x, max(min_y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0), 2)

    for u in unpaired:
        b = u['box']
        cv2.rectangle(annotated, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 0, 255), 2)
        cv2.putText(annotated, "TIDAK BERPASANGAN", (b[0], max(b[1] - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    return annotated, len(candidates), len(pairs), len(unpaired)

def play_audio(audio_path):
    def _play():
        try:
            if os.path.exists(audio_path):
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

def test_image(image_path, model_path=None, audio_path=None, conf_thresh=0.70):
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
    
    print(f"[INFO] Menguji gambar: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Gagal membaca gambar: {image_path}")
        return

    # Analisis deteksi kotak bounding box fisik sandal & pasangan
    annotated_frame, tot_s, num_p, num_u = analyze_and_draw_shoe_pairs(frame)

    results = model.predict(source=frame, verbose=False, device='cpu', imgsz=224)
    result = results[0]
    probs = result.probs
    names = result.names
    
    top1_idx = probs.top1
    raw_label = names[top1_idx].upper()
    confidence = float(probs.top1conf)
    
    print("\n" + "="*50)
    print(f" DETEKSI KOTAK PASANGAN: Total={tot_s}, Pasang={num_p}, Tercecer={num_u}")
    print(f" HASIL KLASIFIKASI MODEL: {raw_label} ({confidence*100:.2f}%)")
    print("="*50)
    for idx, name in names.items():
        print(f" - {name.upper()}: {float(probs.data[idx])*100:.2f}%")
        
    # Keputusan Pasti: Hanya RAPI atau BERANTAKAN (atau KOSONG)
    if tot_s == 0 and raw_label == "KOSONG":
        final_label = "KOSONG"
    elif num_u > 0 or raw_label == "BERANTAKAN":
        final_label = "BERANTAKAN"
    else:
        final_label = "RAPI"
    
    if final_label == "BERANTAKAN":
        banner_color = (0, 0, 220) # Red
        status_text = f"PERINGATAN! BERANTAKAN"
        sub_text = f"Terdeteksi {num_u} sandal tercecer / tidak berpasangan!"
        print("\n[AUDIO] Memutar peringatan suara...")
        play_audio(audio_path)
    elif final_label == "RAPI":
        banner_color = (0, 200, 0) # Green
        status_text = f"STATUS: RAPI"
        sub_text = f"{num_p} pasang sandal/sepatu tertata rapi berdampingan"
    else: # KOSONG
        banner_color = (220, 140, 20) # Blue-cyan (BGR)
        status_text = f"STATUS: KOSONG / STANDBY"
        sub_text = "Area lantai bersih (tidak ada sandal atau sepatu)"

    # Overlay banner on annotated_frame
    h, w, _ = annotated_frame.shape
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        annotated_frame = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        h, w, _ = annotated_frame.shape
        
    overlay = annotated_frame.copy()
    banner_height = max(70, int(h * 0.12))
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    
    font_scale_main = max(0.6, w / 900.0)
    font_scale_sub = max(0.45, w / 1200.0)
    
    cv2.putText(annotated_frame, status_text, (15, int(banner_height * 0.45)), cv2.FONT_HERSHEY_DUPLEX, font_scale_main, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, sub_text, (15, int(banner_height * 0.82)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255, 255, 255), 1, cv2.LINE_AA)
    
    win_title = f"Uji Foto Sandal - {os.path.basename(image_path)}"
    cv2.imshow(win_title, annotated_frame)
    print("\nTekan sembarang tombol pada jendela gambar untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uji Foto Model Deteksi Sandal")
    parser.add_argument("--image", default=None, help="Path ke file foto sandal/sepatu")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path bobot model")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Path audio file")
    parser.add_argument("--conf", type=float, default=0.70, help="Confidence threshold")
    args = parser.parse_args()

    img_file = args.image
    if not img_file:
        print("[PROMPT] Membuka jendela pemilihan file...")
        img_file = pick_image_gui()

    if img_file:
        test_image(img_file, model_path=args.model, audio_path=args.audio, conf_thresh=args.conf)
    else:
        print("[INFO] Tidak ada file gambar yang dipilih.")
