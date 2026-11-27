import os
import sys
import time
import argparse
import threading
import cv2
from PIL import Image
import torch
from ultralytics import YOLO

# Batasi thread CPU PyTorch agar ringan
torch.set_num_threads(2)

# Suppress pygame prompt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np

class MotionSettlementWatcher:
    """
    Detektor Pergerakan Ultra-Ringan (240x160):
    Mendeteksi pergerakan orang saat menaruh sandal di area lantai / rak.
    Capture otomatis dievaluasi HANYA SETELAH pergerakan orang selesai dan kondisi kembali tenang (settled).
    """
    def __init__(self, settle_sec=1.8, motion_thresh=0.025, cooldown_sec=10.0):
        self.settle_sec = settle_sec
        self.motion_thresh = motion_thresh
        self.cooldown_sec = cooldown_sec
        self.prev_gray = None
        self.state = "STANDBY"
        self.last_motion_time = 0.0
        self.motion_start_time = 0.0
        self.last_trigger_time = 0.0
        self.activity_detected = False
        self.motion_score = 0.0

    def update(self, frame_bgr):
        now = time.time()
        h, w = frame_bgr.shape[:2]
        
        # Ambil area lantai (60% bawah) lalu downscale ke 240x160 agar super enteng
        roi = frame_bgr[int(h * 0.40):, :]
        small = cv2.resize(roi, (240, 160), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return "STANDBY", False, 0.0

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        motion_score = np.count_nonzero(thresh) / float(thresh.size)
        self.motion_score = motion_score

        self.prev_gray = gray.copy()

        if self.state == "COOLDOWN":
            if now - self.last_trigger_time >= self.cooldown_sec:
                self.state = "STANDBY"
                self.activity_detected = False
            return "COOLDOWN", False, motion_score

        is_moving = motion_score > self.motion_thresh

        if is_moving:
            self.last_motion_time = now
            if not self.activity_detected:
                self.activity_detected = True
                self.motion_start_time = now
            self.state = "PERSON_MOVING"
            return "PERSON_MOVING", False, motion_score

        if self.activity_detected:
            if now - self.last_motion_time >= self.settle_sec:
                self.activity_detected = False
                self.state = "COOLDOWN"
                self.last_trigger_time = now
                return "TRIGGER_SNAPSHOT", True, motion_score
            else:
                self.state = "SETTLING"
                return "SETTLING", False, motion_score

        self.state = "STANDBY"
        return "STANDBY", False, motion_score

class RealtimeShoeDetector:
    def __init__(self, model_path="models/best.pt", audio_path="audio/peringatan.mp3"):
        self.model_path = model_path
        self.audio_path = audio_path
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {self.model_path}")
            
        print(f"Memuat model YOLOv8 dari: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Warmup model
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self.model.predict(source=dummy, verbose=False, device='cpu', imgsz=224)
        
        # Inisialisasi Audio
        self.audio_enabled = False
        if os.path.exists(self.audio_path):
            try:
                pygame.mixer.init()
                self.audio_enabled = True
                print(f"Modul Audio Aktif: {self.audio_path}")
            except Exception as e:
                print(f"Gagal inisialisasi audio: {e}")
        else:
            print(f"File audio tidak ditemukan di: {self.audio_path}")
            
        self.is_playing_audio = False
        self.watcher = MotionSettlementWatcher(settle_sec=1.8, motion_thresh=0.025, cooldown_sec=10.0)

    def play_warning_audio(self):
        if not self.audio_enabled or self.is_playing_audio:
            return
            
        def _play():
            self.is_playing_audio = True
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(self.audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error memutar audio: {e}")
            finally:
                self.is_playing_audio = False
                
        threading.Thread(target=_play, daemon=True).start()

    def predict_ensemble(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        crop_top = int(h * 0.35)
        floor_crop = img_pil.crop((0, crop_top, w, h))
        flip_img = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Batch inference
        batch_results = self.model.predict(source=[img_pil, floor_crop, flip_img], verbose=False, device='cpu', imgsz=224)
        names = batch_results[0].names
        
        p_global = {names[i].upper(): float(batch_results[0].probs.data[i]) for i in range(len(names))}
        p_crop = {names[i].upper(): float(batch_results[1].probs.data[i]) for i in range(len(names))}
        p_flip = {names[i].upper(): float(batch_results[2].probs.data[i]) for i in range(len(names))}
        
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
        top_label = max(p_ens, key=p_ens.get)
        top_conf = p_ens[top_label]
        
        return top_label, top_conf, p_ens

    def run(self, source=0):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka sumber video '{source}'")
            return
            
        print("\n=== SISTEM DETEKSI SANDAL / SEPATU REAL-TIME AKTIF ===")
        print("Mode: Smart Event-Driven (Capture Otomatis Setelah Orang Menaruh Sandal)")
        print("Tekan 'q' pada jendela video untuk keluar, atau 's' untuk snapshot manual.\n")
        
        prev_time = time.time()
        fps = 0.0
        last_eval_label = "STANDBY"
        last_eval_conf = 0.0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Video stream berakhir atau kamera terputus.")
                break
                
            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(current_time - prev_time, 1e-5))
            prev_time = current_time

            if frame.shape[1] > 640:
                scale = 640.0 / frame.shape[1]
                frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

            h, w = frame.shape[:2]
            annotated = frame.copy()

            # Boundary Drop Zone
            roi_y = int(h * 0.35)
            cv2.rectangle(annotated, (5, roi_y), (w - 5, h - 5), (255, 180, 0), 1)

            # Update Event Watcher
            state, triggered, m_score = self.watcher.update(frame)

            if state == "STANDBY":
                banner_color = (180, 180, 40)
                status_text = "STANDBY: MEMANTAU AREA"
                sub_text = f"Menunggu orang datang | Gerakan: {m_score*100:.1f}%"
            elif state == "PERSON_MOVING":
                banner_color = (0, 165, 255)
                status_text = "AKTIVITAS ORANG TERDETEKSI"
                sub_text = "Sedang meletakkan sandal... Menunggu selesai"
            elif state == "SETTLING":
                banner_color = (255, 140, 0)
                rem_s = max(0.0, self.watcher.settle_sec - (current_time - self.watcher.last_motion_time))
                status_text = f"ORANG SELESAI ({rem_s:.1f}s)"
                sub_text = "Menunggu kondisi tenang sebelum evaluasi..."
            elif state == "COOLDOWN":
                rem_cd = int(max(0.0, self.watcher.cooldown_sec - (current_time - self.watcher.last_trigger_time)))
                banner_color = (120, 120, 120)
                status_text = f"COOLDOWN ({rem_cd}s)"
                sub_text = f"Status Terakhir: {last_eval_label} ({last_eval_conf*100:.1f}%)"

            if triggered:
                print(f"[{time.strftime('%H:%M:%S')}] Event Terdeteksi: Orang selesai menaruh sandal! Mengevaluasi...")
                top_label, top_conf, p_ens = self.predict_ensemble(frame)
                last_eval_label = top_label
                last_eval_conf = top_conf
                
                print(f"[{time.strftime('%H:%M:%S')}] Hasil Ensemble: {top_label} ({top_conf*100:.1f}%) - {p_ens}")
                
                if top_label == "BERANTAKAN":
                    print(f"[{time.strftime('%H:%M:%S')}] Trigger Suara Peringatan: Sandal Berantakan!")
                    self.play_warning_audio()

            # Render Overlay Banner
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), banner_color, -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            cv2.putText(annotated, status_text, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"{sub_text} | FPS: {fps:.1f}", (15, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Smart Shoe Monitor - Lab TI (Real-Time)", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("Snapshot manual dipicu...")
                top_label, top_conf, p_ens = self.predict_ensemble(frame)
                last_eval_label = top_label
                last_eval_conf = top_conf
                if top_label == "BERANTAKAN":
                    self.play_warning_audio()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi Kerapian Sandal Real-Time Event-Driven")
    parser.add_argument("--source", default=0, help="Sumber kamera: index integer webcam (0/1) atau URL stream IP Camera")
    parser.add_argument("--model", default="models/best.pt", help="Path bobot model YOLO")
    parser.add_argument("--audio", default="audio/peringatan.mp3", help="Path file suara peringatan")
    args = parser.parse_args()

    detector = RealtimeShoeDetector(model_path=args.model, audio_path=args.audio)
    detector.run(source=args.source)
