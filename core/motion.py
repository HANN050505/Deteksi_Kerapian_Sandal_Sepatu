"""
Modul pemantau pergerakan dan penenangan kondisi (Motion Settlement Watcher).
Menjadi satu-satunya sumber kebenaran (single source of truth) untuk deteksi pergerakan
sebelum capture otomatis dievaluasi oleh model deep learning.
"""
from __future__ import annotations

from enum import Enum
import time
from typing import Literal

import cv2
import numpy as np

from core.config import (
    FLOOR_ROI_TOP_RATIO,
    MOTION_COOLDOWN_SECONDS,
    MOTION_DIFF_THRESHOLD,
    MOTION_DOWNSCALE_SIZE,
    MOTION_SCORE_THRESHOLD,
    MOTION_SETTLE_SECONDS,
)


class DetectorState(str, Enum):
    """Enumerasi state eksplisit detektor pergerakan."""
    STANDBY = "STANDBY"
    PERSON_MOVING = "PERSON_MOVING"
    SETTLING = "SETTLING"
    TRIGGER_SNAPSHOT = "TRIGGER_SNAPSHOT"
    COOLDOWN = "COOLDOWN"


WatcherState = Literal["STANDBY", "PERSON_MOVING", "SETTLING", "TRIGGER_SNAPSHOT", "COOLDOWN"]


class MotionSettlementWatcher:
    """Detektor pergerakan ultra-ringan berbasis downscaling dan background difference.

    Memantau area lantai yang telah disepakati (FLOOR_ROI_TOP_RATIO) dan hanya memicu
    sinyal evaluasi (triggered=True) saat orang telah selesai meletakkan sandal dan
    kondisi area lantai kembali tenang (settled).
    """

    def __init__(
        self,
        settle_sec: float = MOTION_SETTLE_SECONDS,
        motion_thresh: float = MOTION_SCORE_THRESHOLD,
        cooldown_sec: float = MOTION_COOLDOWN_SECONDS,
        roi_top_ratio: float = FLOOR_ROI_TOP_RATIO,
    ) -> None:
        """Inisialisasi watcher pergerakan."""
        self.settle_sec: float = float(settle_sec)
        self.motion_thresh: float = float(motion_thresh)
        self.cooldown_sec: float = float(cooldown_sec)
        self.roi_top_ratio: float = float(roi_top_ratio)

        self.prev_gray: np.ndarray | None = None
        self.state: DetectorState = DetectorState.STANDBY
        self.last_motion_time: float = 0.0
        self.motion_start_time: float = 0.0
        self.last_trigger_time: float = 0.0
        self.activity_detected: bool = False
        self.motion_during_cooldown: bool = False
        self.motion_score: float = 0.0

    def reset(self) -> None:
        """Mereset kondisi internal watcher."""
        self.prev_gray = None
        self.state = DetectorState.STANDBY
        self.last_motion_time = 0.0
        self.motion_start_time = 0.0
        self.last_trigger_time = 0.0
        self.activity_detected = False
        self.motion_during_cooldown = False
        self.motion_score = 0.0

    def update(self, frame_bgr: np.ndarray) -> tuple[str, bool, float]:
        """Memproses satu frame citra BGR dan memperbarui status aktivitas.

        Args:
            frame_bgr: Citra masukan dari kamera dalam format BGR OpenCV.

        Returns:
            Tuple berisi:
            - state: Status saat ini ("STANDBY", "PERSON_MOVING", "SETTLING", "TRIGGER_SNAPSHOT", "COOLDOWN").
            - triggered: True jika kondisi tepat untuk memicu snapshot evaluasi.
            - motion_score: Skor intensitas pergerakan piksel saat ini (0.0 - 1.0).
        """
        now = time.time()
        h, _ = frame_bgr.shape[:2]

        # Ambil area lantai seragam menggunakan FLOOR_ROI_TOP_RATIO
        crop_y = int(h * self.roi_top_ratio)
        roi = frame_bgr[crop_y:, :]

        # Downscale ke resolusi ringkas agar inferensi pergerakan < 0.2 ms
        small = cv2.resize(roi, MOTION_DOWNSCALE_SIZE, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return DetectorState.STANDBY.value, False, 0.0

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion_score = float(np.count_nonzero(thresh) / float(thresh.size))
        self.motion_score = motion_score
        self.prev_gray = gray.copy()

        is_moving = motion_score > self.motion_thresh

        # Handle masa Cooldown (Fase 2: Catat gerakan selama cooldown, lalu evaluasi setelah cooldown)
        if self.state == DetectorState.COOLDOWN:
            if is_moving:
                self.motion_during_cooldown = True
                self.last_motion_time = now

            if now - self.last_trigger_time >= self.cooldown_sec:
                # Masa cooldown berakhir
                if self.motion_during_cooldown or is_moving:
                    # Terjadi pergerakan saat cooldown, evaluasi ulang setelah kondisi stabil
                    self.activity_detected = True
                    self.motion_during_cooldown = False
                    if is_moving:
                        self.state = DetectorState.PERSON_MOVING
                        return DetectorState.PERSON_MOVING.value, False, motion_score
                    else:
                        self.state = DetectorState.SETTLING
                        return DetectorState.SETTLING.value, False, motion_score
                else:
                    self.state = DetectorState.STANDBY
                    self.activity_detected = False
                    return DetectorState.STANDBY.value, False, motion_score
            else:
                return DetectorState.COOLDOWN.value, False, motion_score

        if is_moving:
            self.last_motion_time = now
            if not self.activity_detected:
                self.activity_detected = True
                self.motion_start_time = now
            self.state = DetectorState.PERSON_MOVING
            return DetectorState.PERSON_MOVING.value, False, motion_score

        # Jika tidak ada gerakan, periksa apakah baru saja ada aktivitas
        if self.activity_detected:
            quiet_duration = now - self.last_motion_time
            if quiet_duration >= self.settle_sec:
                # Kondisi tenang tercapai setelah aktivitas: trigger snapshot!
                self.activity_detected = False
                self.state = DetectorState.COOLDOWN
                self.last_trigger_time = now
                self.motion_during_cooldown = False
                return DetectorState.TRIGGER_SNAPSHOT.value, True, motion_score
            self.state = DetectorState.SETTLING
            return DetectorState.SETTLING.value, False, motion_score

        self.state = DetectorState.STANDBY
        return DetectorState.STANDBY.value, False, motion_score
