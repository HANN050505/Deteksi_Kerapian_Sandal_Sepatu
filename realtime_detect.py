"""
Aplikasi deteksi kerapian sandal/sepatu secara real-time via webcam atau video stream.
Menggunakan arsitektur Event-Driven Motion Settlement Watcher dan modul terpadu core.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import pygame

from core import (
    DEFAULT_AUDIO_PATH,
    DEFAULT_MODEL_PATH,
    FLOOR_ROI_TOP_RATIO,
    MOTION_COOLDOWN_SECONDS,
    MOTION_SCORE_THRESHOLD,
    MOTION_SETTLE_SECONDS,
    MotionSettlementWatcher,
    load_model,
    predict_ensemble,
)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


class RealtimeShoeDetector:
    """Detektor kerapian sandal/sepatu berbasis video stream real-time."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        audio_path: str | Path = DEFAULT_AUDIO_PATH,
        roi_top_ratio: float = FLOOR_ROI_TOP_RATIO,
    ) -> None:
        """Inisialisasi detektor real-time, model YOLO, dan sub-sistem audio."""
        self.model_path = Path(model_path)
        self.audio_path = Path(audio_path)
        self.roi_top_ratio = float(roi_top_ratio)

        print(f"Memuat model YOLOv8 dari: {self.model_path}")
        self.model = load_model(self.model_path)

        # Inisialisasi Audio Mixer
        self.audio_enabled = False
        if self.audio_path.exists():
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                self.audio_enabled = True
                print(f"Modul Audio Aktif: {self.audio_path}")
            except Exception as e:
                print(f"Gagal inisialisasi audio: {e}")
        else:
            print(f"File audio tidak ditemukan di: {self.audio_path}")

        self.is_playing_audio = False
        self.watcher = MotionSettlementWatcher(
            settle_sec=MOTION_SETTLE_SECONDS,
            motion_thresh=MOTION_SCORE_THRESHOLD,
            cooldown_sec=MOTION_COOLDOWN_SECONDS,
            roi_top_ratio=self.roi_top_ratio,
        )

    def play_warning_audio(self) -> None:
        """Memutar berkas audio peringatan secara non-blocking di latar belakang."""
        if not self.audio_enabled or self.is_playing_audio:
            return

        def _play() -> None:
            self.is_playing_audio = True
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(str(self.audio_path))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error pemutaran audio: {e}")
            finally:
                self.is_playing_audio = False

        threading.Thread(target=_play, daemon=True).start()

    def run(self, source: str | int = 0) -> None:
        """Menjalankan loop penangkapan frame video stream dan evaluasi aktivitas."""
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY)
        if not cap.isOpened():
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

            # Boundary Drop Zone pemantauan lantai (seragam dengan konstanta)
            roi_y = int(h * self.roi_top_ratio)
            cv2.rectangle(annotated, (5, roi_y), (w - 5, h - 5), (255, 180, 0), 1)

            # Inisialisasi default banner
            banner_color: tuple[int, int, int] = (180, 180, 40)
            status_text: str = "STANDBY: MEMANTAU AREA"
            sub_text: str = f"Status: {last_eval_label}"

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
                pred = predict_ensemble(self.model, frame, roi_top_ratio=self.roi_top_ratio)
                last_eval_label = pred["label"]
                last_eval_conf = pred["confidence"]

                print(
                    f"[{time.strftime('%H:%M:%S')}] Hasil Ensemble: {last_eval_label} "
                    f"({last_eval_conf*100:.1f}%) - {pred['probs']}"
                )

                if last_eval_label == "BERANTAKAN":
                    print(f"[{time.strftime('%H:%M:%S')}] Trigger Suara Peringatan: Sandal Berantakan!")
                    self.play_warning_audio()

            # Render Overlay Banner
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), banner_color, -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            cv2.putText(
                annotated,
                status_text,
                (15, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"{sub_text} | FPS: {fps:.1f}",
                (15, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Smart Shoe Monitor - Lab TI (Real-Time)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                print("Snapshot manual dipicu...")
                pred = predict_ensemble(self.model, frame, roi_top_ratio=self.roi_top_ratio)
                last_eval_label = pred["label"]
                last_eval_conf = pred["confidence"]
                if last_eval_label == "BERANTAKAN":
                    self.play_warning_audio()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi Kerapian Sandal Real-Time Event-Driven")
    parser.add_argument("--source", default=0, help="Sumber kamera: index integer webcam (0/1) atau URL stream")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path bobot model YOLO")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO_PATH), help="Path file suara peringatan")
    args = parser.parse_args()

    detector = RealtimeShoeDetector(model_path=args.model, audio_path=args.audio)
    detector.run(source=args.source)
