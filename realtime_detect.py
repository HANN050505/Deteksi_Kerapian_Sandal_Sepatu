"""
Aplikasi deteksi kerapian sandal/sepatu secara real-time via webcam atau video stream.
Menggunakan arsitektur Event-Driven Motion Settlement Watcher, threaded asynchronous inference,
temporal voting filter, dan penanganan state mesin lengkap bebas latency freeze.
"""
from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
import pygame

from core import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_AUDIO_PATH,
    DEFAULT_MODEL_PATH,
    DetectorState,
    ENSEMBLE_ENABLED,
    FLOOR_ROI_TOP_RATIO,
    MOTION_COOLDOWN_SECONDS,
    MOTION_SCORE_THRESHOLD,
    MOTION_SETTLE_SECONDS,
    PERIODIC_EVAL_SECONDS,
    PERIODIC_RECHECK_SECONDS,
    TEMPORAL_VOTING_MIN_HITS,
    TEMPORAL_VOTING_WINDOW,
    USE_LETTERBOX_PREPROCESSING,
    MotionSettlementWatcher,
    load_model,
    predict_ensemble,
    predict_single,
)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


class RealtimeShoeDetector:
    """Detektor kerapian sandal/sepatu real-time dengan threaded async inference."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        audio_path: str | Path = DEFAULT_AUDIO_PATH,
        roi_top_ratio: float = FLOOR_ROI_TOP_RATIO,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        enable_ensemble: bool = ENSEMBLE_ENABLED,
        temporal_window: int = TEMPORAL_VOTING_WINDOW,
        temporal_min_hits: int = TEMPORAL_VOTING_MIN_HITS,
        periodic_eval_sec: float = PERIODIC_EVAL_SECONDS,
        use_letterbox: bool = USE_LETTERBOX_PREPROCESSING,
    ) -> None:
        """Inisialisasi detektor real-time, model YOLO, audio mixer, dan antrean inferensi terpisah."""
        self.model_path = Path(model_path)
        self.audio_path = Path(audio_path)
        self.roi_top_ratio = float(roi_top_ratio)
        self.confidence_threshold = float(confidence_threshold)
        self.enable_ensemble = bool(enable_ensemble)
        self.temporal_window = int(temporal_window)
        self.temporal_min_hits = int(temporal_min_hits)
        self.periodic_eval_sec = float(periodic_eval_sec)
        self.use_letterbox = bool(use_letterbox)

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

        # Buffer temporal voting keputusan (antrean N evaluasi terakhir)
        self.voting_buffer: deque[str] = deque(maxlen=self.temporal_window)

        # Threaded Asynchronous Inference Queue
        # Memisahkan eksekusi model dari loop video agar feed kamera tidak mengalami freeze/stutter
        self.inference_queue: queue.Queue[tuple[np.ndarray, str]] = queue.Queue(maxsize=1)
        self.latest_result: dict[str, Any] | None = None
        self.result_lock = threading.Lock()
        self.is_evaluating = False

        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()

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

    def _inference_worker(self) -> None:
        """Pekerja latar belakang (background thread) untuk menjalankan inferensi YOLO."""
        while True:
            try:
                frame_to_eval, trigger_source = self.inference_queue.get()
                self.is_evaluating = True

                t_start = time.time()
                if self.enable_ensemble:
                    pred = predict_ensemble(
                        self.model,
                        frame_to_eval,
                        roi_top_ratio=self.roi_top_ratio,
                        confidence_threshold=self.confidence_threshold,
                        use_letterbox=self.use_letterbox,
                    )
                else:
                    pred = predict_single(
                        self.model,
                        frame_to_eval,
                        confidence_threshold=self.confidence_threshold,
                        use_letterbox=self.use_letterbox,
                    )
                eval_latency_ms = (time.time() - t_start) * 1000.0

                label = pred["label"]
                conf = pred["confidence"]

                # Simpan ke riwayat voting temporal
                self.voting_buffer.append(label)
                berantakan_hits = self.voting_buffer.count("BERANTAKAN")

                pred["eval_latency_ms"] = eval_latency_ms
                pred["trigger_source"] = trigger_source
                pred["berantakan_hits"] = berantakan_hits

                with self.result_lock:
                    self.latest_result = pred

                print(
                    f"[{time.strftime('%H:%M:%S')}] Evaluasi ({trigger_source}): {label} ({conf*100:.1f}%) "
                    f"| Latensi: {eval_latency_ms:.1f}ms | Voting: {berantakan_hits}/{len(self.voting_buffer)} Hits"
                )

                # Evaluasi syarat pemutaran audio:
                # 1. Label bukan 'RAGU' (confidence >= threshold)
                # 2. Minimal K dari N evaluasi berurutan bernilai 'BERANTAKAN'
                if label == "BERANTAKAN" and berantakan_hits >= self.temporal_min_hits:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Peringatan Terverifikasi ({berantakan_hits}/"
                        f"{self.temporal_window}): Memutar audio!"
                    )
                    self.play_warning_audio()

                self.is_evaluating = False
                self.inference_queue.task_done()
            except Exception as e:
                print(f"Error pada thread inferensi: {e}")
                self.is_evaluating = False

    def request_evaluation(self, frame: np.ndarray, source_tag: str = "EVENT") -> bool:
        """Mengirim frame untuk dievaluasi oleh thread latar belakang tanpa blocking."""
        if self.inference_queue.full():
            return False
        try:
            self.inference_queue.put_nowait((frame.copy(), source_tag))
            return True
        except queue.Full:
            return False

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

        print("\n=== SISTEM DETEKSI SANDAL / SEPATU REAL-TIME (FASE 2) ===")
        print(f"Mode: Threaded Async | Ensemble: {'AKTIF' if self.enable_ensemble else 'NONAKTIF'}")
        print(f"Threshold Keyakinan: {self.confidence_threshold*100:.0f}% | Temporal Voting: {self.temporal_min_hits}/{self.temporal_window}")
        print("Tekan 'q' pada jendela video untuk keluar, atau 's' untuk snapshot manual.\n")

        prev_time = time.time()
        fps = 0.0
        last_eval_time = time.time()

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

            # Inisialisasi variabel status dan banner di awal loop (menjamin bebas NameError)
            banner_color: tuple[int, int, int] = (180, 180, 40)
            status_text: str = "STANDBY: MEMANTAU AREA"
            sub_text: str = "Menunggu aktivitas..."

            # Ambil hasil evaluasi terakhir jika tersedia
            with self.result_lock:
                latest = self.latest_result

            latest_label = latest["label"] if latest else "STANDBY"
            latest_conf = latest["confidence"] if latest else 0.0
            hits_count = latest.get("berantakan_hits", 0) if latest else 0

            # Update Event Watcher (Fase 2: Catat gerakan saat cooldown & dukung semua state)
            state_str, triggered, m_score = self.watcher.update(frame)

            # Penanganan seluruh state secara lengkap melalui Enum
            if state_str == DetectorState.STANDBY.value:
                banner_color = (180, 180, 40)
                status_text = "STANDBY: MEMANTAU AREA"
                sub_text = f"Menunggu orang datang | Gerakan: {m_score*100:.1f}%"

                # Evaluasi periodik saat STANDBY agar kondisi berantakan sejak awal tetap terdeteksi
                if current_time - last_eval_time >= self.periodic_eval_sec:
                    last_eval_time = current_time
                    self.request_evaluation(frame, source_tag="PERIODIC_STANDBY")

            elif state_str == DetectorState.PERSON_MOVING.value:
                banner_color = (0, 165, 255)
                status_text = "AKTIVITAS ORANG TERDETEKSI"
                sub_text = "Sedang meletakkan sandal... Menunggu selesai"

            elif state_str == DetectorState.SETTLING.value:
                banner_color = (255, 140, 0)
                rem_s = max(0.0, self.watcher.settle_sec - (current_time - self.watcher.last_motion_time))
                status_text = f"ORANG SELESAI ({rem_s:.1f}s)"
                sub_text = "Menunggu kondisi tenang sebelum evaluasi..."

            elif state_str == DetectorState.TRIGGER_SNAPSHOT.value or triggered:
                banner_color = (200, 0, 200)
                status_text = "TRIGGER SNAPSHOT: EVALUASI AI"
                sub_text = "Mengevaluasi kerapian di background thread..."
                last_eval_time = current_time
                self.request_evaluation(frame, source_tag="MOTION_SETTLED")

            elif state_str == DetectorState.COOLDOWN.value:
                rem_cd = int(max(0.0, self.watcher.cooldown_sec - (current_time - self.watcher.last_trigger_time)))
                banner_color = (120, 120, 120)
                status_text = f"COOLDOWN ({rem_cd}s)"
                sub_text = f"Hasil Terakhir: {latest_label} ({latest_conf*100:.1f}%) | Voting: {hits_count}/{self.temporal_window}"

                # Re-check periodik saat cooldown jika sebelumnya terdeteksi berantakan
                if latest_label == "BERANTAKAN" and (current_time - last_eval_time >= PERIODIC_RECHECK_SECONDS):
                    last_eval_time = current_time
                    self.request_evaluation(frame, source_tag="PERIODIC_RECHECK")

            # Render Status Evaluasi Terbaru di Sudut Kiri Bawah
            if latest:
                lbl = latest["label"]
                cnf = latest["confidence"] * 100
                res_bg = (0, 0, 220) if lbl == "BERANTAKAN" else ((0, 200, 0) if lbl == "RAPI" else ((0, 165, 255) if lbl == "RAGU" else (220, 140, 20)))
                cv2.rectangle(annotated, (8, h - 35), (320, h - 8), res_bg, -1)
                cv2.putText(
                    annotated,
                    f"AI: {lbl} ({cnf:.1f}%) | Voting:{hits_count}/{self.temporal_window}",
                    (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Render Overlay Banner Atas
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
                last_eval_time = current_time
                self.request_evaluation(frame, source_tag="MANUAL_SNAPSHOT")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi Kerapian Sandal Real-Time Event-Driven (Fase 2)")
    parser.add_argument("--source", default=0, help="Sumber kamera: index integer webcam (0/1) atau URL stream")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path bobot model YOLO")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO_PATH), help="Path file suara peringatan")
    parser.add_argument("--confidence-thresh", type=float, default=CONFIDENCE_THRESHOLD, help="Ambang batas keyakinan minimum (default 0.70)")
    parser.add_argument("--no-ensemble", action="store_true", help="Nonaktifkan ensemble, gunakan model tunggal")
    parser.add_argument("--temporal-k", type=int, default=TEMPORAL_VOTING_MIN_HITS, help="Minimal hits voting temporal")
    parser.add_argument("--temporal-n", type=int, default=TEMPORAL_VOTING_WINDOW, help="Ukuran jendela voting temporal")
    parser.add_argument("--periodic-eval", type=float, default=PERIODIC_EVAL_SECONDS, help="Interval evaluasi periodik saat standby (detik)")
    args = parser.parse_args()

    detector = RealtimeShoeDetector(
        model_path=args.model,
        audio_path=args.audio,
        confidence_threshold=args.confidence_thresh,
        enable_ensemble=not args.no_ensemble,
        temporal_window=args.temporal_n,
        temporal_min_hits=args.temporal_k,
        periodic_eval_sec=args.periodic_eval,
    )
    detector.run(source=args.source)
