"""
Unit tests dan script verifikasi otomatis untuk FASE 2:
- State Machine & Motion Watcher (Cooldown motion tracking & TRIGGER_SNAPSHOT)
- Letterbox Preprocessing (Square padding untuk mempertahankan full FOV)
- Confidence Thresholding (Status 'RAGU' & pencegahan false alarm)
- Ensemble On/Off Toggle
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

# Pastikan root direktori proyek ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from PIL import Image

from core import (
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_UNCERTAIN_LABEL,
    DetectorState,
    ENSEMBLE_ENABLED,
    MotionSettlementWatcher,
    letterbox_to_square,
    load_model,
    predict_ensemble,
    predict_single,
)


def test_motion_watcher_state_machine() -> None:
    print("[1] Pengujian State Machine & Cooldown Motion Tracking...")
    watcher = MotionSettlementWatcher(settle_sec=0.2, motion_thresh=0.02, cooldown_sec=0.4)

    # Frame hitam kosong (stabil)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Frame dengan perubahan besar (ada gerakan)
    moving_frame = np.full((480, 640, 3), 200, dtype=np.uint8)

    # 1. Start in STANDBY
    s1, trig1, _ = watcher.update(blank_frame)
    assert s1 == DetectorState.STANDBY.value, f"Expected STANDBY, got {s1}"

    # 2. Movement starts -> PERSON_MOVING
    s2, trig2, _ = watcher.update(moving_frame)
    assert s2 == DetectorState.PERSON_MOVING.value, f"Expected PERSON_MOVING, got {s2}"

    # 3. Movement stops -> SETTLING
    s3, trig3, _ = watcher.update(moving_frame)  # diff is 0 now
    assert s3 == DetectorState.SETTLING.value, f"Expected SETTLING, got {s3}"

    # 4. Wait for settle -> TRIGGER_SNAPSHOT
    time.sleep(0.25)
    s4, trig4, _ = watcher.update(moving_frame)
    assert s4 == DetectorState.TRIGGER_SNAPSHOT.value, f"Expected TRIGGER_SNAPSHOT, got {s4}"
    assert trig4 is True, "Expected triggered == True"

    # 5. Immediately enters COOLDOWN
    s5, trig5, _ = watcher.update(moving_frame)
    assert s5 == DetectorState.COOLDOWN.value, f"Expected COOLDOWN, got {s5}"

    # 6. Motion happens during cooldown (should be remembered!)
    watcher.update(blank_frame)  # creates diff against moving_frame
    assert watcher.motion_during_cooldown is True, "Expected motion_during_cooldown to be True"

    # 7. Wait for cooldown to expire (0.4s)
    time.sleep(0.45)
    s6, trig6, _ = watcher.update(blank_frame)
    # Because motion occurred during cooldown, it must transition to SETTLING (or PERSON_MOVING), NOT blindly to STANDBY
    assert s6 in (DetectorState.SETTLING.value, DetectorState.PERSON_MOVING.value), f"Expected SETTLING/PERSON_MOVING, got {s6}"
    print("    -> PASS: Motion watcher menangani semua state & mencatat pergerakan cooldown dengan benar.")


def test_letterbox_preprocessing() -> None:
    print("[2] Pengujian Preprocessing Letterbox Bujur Sangkar...")
    # Citra lebar 16:9
    rect_img = Image.new("RGB", (640, 360), color=(50, 100, 150))
    sq_img = letterbox_to_square(rect_img)

    assert sq_img.size == (640, 640), f"Expected (640, 640), got {sq_img.size}"
    # Periksa warna padding netral di bagian atas/bawah
    pad_pixel = sq_img.getpixel((320, 10))
    assert pad_pixel == (114, 114, 114), f"Expected padding (114, 114, 114), got {pad_pixel}"
    # Periksa citra asli tetap utuh di tengah
    center_pixel = sq_img.getpixel((320, 320))
    assert center_pixel == (50, 100, 150), f"Expected center (50, 100, 150), got {center_pixel}"
    print("    -> PASS: Letterbox square pad menjaga seluruh bidang pandang tanpa cropping samping.")


def test_inference_thresholding_and_ensemble_toggle() -> None:
    print("[3] Pengujian Inferensi: Model Tunggal vs Ensemble & Threshold Keyakinan...")
    from core import DEFAULT_MODEL_PATH
    model = load_model(DEFAULT_MODEL_PATH)

    sample_img = "dataset/val/BERANTAKAN/new_berantakan_000_24-003 Berantakan 2.jpg"
    assert os.path.exists(sample_img), f"Sample image {sample_img} tidak ditemukan"

    img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    # Uji model tunggal (predict_single)
    res_single = predict_single(model, sample_img)
    assert "label" in res_single
    assert res_single["mode"] == "single"
    print(f"    - Single Model Result : {res_single['label']} ({res_single['confidence']*100:.1f}%)")

    # Uji ensemble (predict_ensemble)
    res_ens = predict_ensemble(model, sample_img, enable_ensemble=True)
    assert res_ens["mode"] == "ensemble"
    print(f"    - Ensemble Result     : {res_ens['label']} ({res_ens['confidence']*100:.1f}%)")

    # Uji toggle enable_ensemble=False
    res_off = predict_ensemble(model, sample_img, enable_ensemble=False)
    assert res_off["mode"] == "single"
    print(f"    - Ensemble Off Toggle : Berhasil fallback ke single ({res_off['label']})")

    # Uji thresholding ketat (confidence_threshold = 0.9999) -> harus menghasilkan 'RAGU'
    res_ragu = predict_single(model, sample_img, confidence_threshold=0.9999)
    assert res_ragu["label"] == CONFIDENCE_UNCERTAIN_LABEL, f"Expected RAGU, got {res_ragu['label']}"
    assert res_ragu["is_confident"] is False
    print(f"    - Thresholding RAGU   : Berhasil mendeteksi RAGU ketika confidence < threshold")
    print("    -> PASS: Logika inferensi, thresholding, dan saklar ensemble bekerja 100% valid.")


if __name__ == "__main__":
    test_motion_watcher_state_machine()
    test_letterbox_preprocessing()
    test_inference_thresholding_and_ensemble_toggle()
    print("\n=== SELURUH TEST VERIFIKASI FASE 2 BERHASIL! ===")
