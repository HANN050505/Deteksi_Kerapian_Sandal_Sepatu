import os
import sys
import time
import argparse
import threading
import cv2
from ultralytics import YOLO

# Suppress pygame prompt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
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

class ShoeMonitor:
    def __init__(self, model_path, audio_path, conf_thresh=0.70, trigger_sec=2.5, cooldown_sec=12.0):
        self.model_path = model_path
        self.audio_path = audio_path
        self.conf_thresh = conf_thresh
        self.trigger_sec = trigger_sec
        self.cooldown_sec = cooldown_sec
        
        # Audio setup
        pygame.mixer.init()
        self.is_playing_audio = False
        
        # State tracking
        self.messy_start_time = None
        self.last_alert_time = 0.0
        
        # Load Model
        print(f"Memuat model deteksi dari: {self.model_path} ...")
        self.model = YOLO(self.model_path)
        print("Model berhasil dimuat!")

    def play_warning_audio(self):
        def _play():
            self.is_playing_audio = True
            try:
                if os.path.exists(self.audio_path):
                    pygame.mixer.music.load(self.audio_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                else:
                    print(f"Audio file tidak ditemukan: {self.audio_path}")
            except Exception as e:
                print(f"Error memutar audio: {e}")
            finally:
                self.is_playing_audio = False
                
        t = threading.Thread(target=_play, daemon=True)
        t.start()

    def run(self, source=0):
        # Handle string integer source e.g. "0"
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka sumber video '{source}'")
            return
            
        print("\n=== SISTEM DETEKSI SANDAL / SEPATU REAL-TIME AKTIF ===")
        print("Tekan 'q' pada jendela video untuk keluar.\n")
        
        prev_time = time.time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video stream berakhir atau kamera terputus.")
                break
                
            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(current_time - prev_time, 1e-5))
            prev_time = current_time
            
            # Analisis deteksi kotak bounding box fisik sandal & pasangan
            annotated_frame, tot_s, num_p, num_u = analyze_and_draw_shoe_pairs(frame)

            # Predict with YOLO
            results = self.model.predict(source=frame, verbose=False, device='cpu', imgsz=224)
            result = results[0]
            
            # Extract top-1 class & confidence
            probs = result.probs
            top1_idx = probs.top1
            raw_label = result.names[top1_idx].upper()
            confidence = float(probs.top1conf)
            
            # Keputusan Pasti: Hanya RAPI atau BERANTAKAN (atau KOSONG)
            if tot_s == 0 and raw_label == "KOSONG":
                label = "KOSONG"
            elif num_u > 0 or raw_label == "BERANTAKAN":
                label = "BERANTAKAN"
            else:
                label = "RAPI"
            
            # Logic Peringatan & Status
            time_since_last_alert = current_time - self.last_alert_time
            in_cooldown = time_since_last_alert < self.cooldown_sec

            if label == "BERANTAKAN":
                if in_cooldown:
                    banner_color = (255, 140, 0) # Orange (Cooldown)
                    remaining = int(self.cooldown_sec - time_since_last_alert)
                    status_text = f"COOLDOWN: {remaining}s"
                    sub_text = f"Tercecer: {num_u} | Peringatan suara dijeda"
                else:
                    if self.messy_start_time is None:
                        self.messy_start_time = current_time
                        
                    duration = current_time - self.messy_start_time
                    
                    if duration >= self.trigger_sec:
                        banner_color = (0, 0, 220) # Red
                        status_text = "PERINGATAN! SANDAL BERANTAKAN"
                        sub_text = f"{num_u} Tercecer / Tidak Berpasangan | Alarm Bunyi!"
                        
                        # Trigger Audio
                        if not self.is_playing_audio:
                            print(f"[{time.strftime('%H:%M:%S')}] Trigger Suara: Sandal berantakan terdeteksi ({num_u} tercecer)")
                            self.play_warning_audio()
                            self.last_alert_time = current_time
                            self.messy_start_time = None
                    else:
                        banner_color = (0, 165, 255) # Yellow-Orange
                        status_text = f"TERDETEKSI BERANTAKAN ({duration:.1f}s / {self.trigger_sec:.1f}s)"
                        sub_text = f"Tercecer: {num_u} sandal | Memverifikasi..."
            else:
                self.messy_start_time = None
                if label == "KOSONG":
                    banner_color = (220, 140, 20) # Blue-Cyan (BGR)
                    status_text = "STATUS: KOSONG / STANDBY"
                    sub_text = "Area lantai bersih / tidak ada sandal"
                else: # RAPI
                    banner_color = (0, 200, 0) # Green
                    status_text = "STATUS: RAPI"
                    sub_text = f"{num_p} Pasang Sandal Tertata Rapi Berdampingan"

            # Draw UI Overlay on annotated_frame (yang sudah memiliki kotak hijau/merah)
            h, w, _ = annotated_frame.shape
            
            # Semi-transparent top bar
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 85), banner_color, -1)
            alpha = 0.75
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
            # Text information
            cv2.putText(annotated_frame, status_text, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, sub_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            
            # FPS indicator in bottom corner
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (w - 120, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_frame, "Tekan 'q' untuk keluar", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Smart Shoe Monitor - Lab Teknik Informatika (Real-Time)", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("Aplikasi ditutup.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(BASE_DIR, "models", "best.pt")
    default_audio = os.path.join(BASE_DIR, "audio", "peringatan.mp3")

    parser = argparse.ArgumentParser(description="Smart Shoe Monitor Real-Time")
    parser.add_argument("--source", default="0", help="Kamera ID (0, 1) atau path file video")
    parser.add_argument("--model", default=default_model, help="Path model weights")
    parser.add_argument("--audio", default=default_audio, help="Path audio file")
    parser.add_argument("--conf", type=float, default=0.70, help="Confidence threshold")
    parser.add_argument("--trigger", type=float, default=2.5, help="Durasi berantakan sebelum bersuara (detik)")
    parser.add_argument("--cooldown", type=float, default=12.0, help="Jeda waktu antar peringatan suara (detik)")
    
    args = parser.parse_args()
    
    monitor = ShoeMonitor(
        model_path=args.model,
        audio_path=args.audio,
        conf_thresh=args.conf,
        trigger_sec=args.trigger,
        cooldown_sec=args.cooldown
    )
    monitor.run(source=args.source)
