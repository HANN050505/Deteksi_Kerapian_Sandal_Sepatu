"""
Modul inferensi Deep Learning (prediksi tunggal dan Test-Time Spatial Ensemble).
Menjadi satu-satunya sumber implementasi evaluasi YOLOv8 classifier pada proyek.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from core.config import (
    CLASSES,
    DEFAULT_MODEL_PATH,
    ENSEMBLE_ENABLED,
    ENSEMBLE_WEIGHT_CROP,
    ENSEMBLE_WEIGHT_FLIP,
    ENSEMBLE_WEIGHT_GLOBAL,
    FLOOR_ROI_TOP_RATIO,
    IMG_SIZE,
)

# Batasi thread PyTorch CPU agar efisien dan tidak membebani prosesor
torch.set_num_threads(2)


def load_model(model_path: str | Path | None = None) -> YOLO:
    """Memuat bobot model YOLOv8 Classifier dan melakukan pemanasan (warmup).

    Args:
        model_path: Path file bobot model (.pt). Jika None, memakai DEFAULT_MODEL_PATH.

    Returns:
        Instance model Ultralytics YOLO yang telah siap diinferensi.
    """
    resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan di: {resolved_path}")

    model = YOLO(str(resolved_path))

    # Pemanasan model awal (warmup) dengan dummy frame
    try:
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        model.predict(source=dummy, verbose=False, device="cpu", imgsz=IMG_SIZE)
    except Exception:
        pass

    return model


def _to_pil_and_bgr(
    img_pil_or_bgr: Image.Image | np.ndarray,
) -> tuple[Image.Image, np.ndarray]:
    """Mengonversi masukan citra fleksibel menjadi pasangan (PIL.Image, BGR np.ndarray)."""
    if isinstance(img_pil_or_bgr, Image.Image):
        img_rgb = img_pil_or_bgr.convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        return img_rgb, frame_bgr
    if isinstance(img_pil_or_bgr, np.ndarray):
        frame_bgr = img_pil_or_bgr
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb), frame_bgr
    raise TypeError(f"Tipe citra tidak didukung: {type(img_pil_or_bgr)}")


def predict_single(
    model: YOLO,
    img_pil_or_bgr: Image.Image | np.ndarray,
    img_size: int = IMG_SIZE,
) -> dict[str, Any]:
    """Menjalankan prediksi tunggal (single perspective) pada satu citra utuh.

    Args:
        model: Instance model YOLOv8.
        img_pil_or_bgr: Citra masukan (PIL Image atau NumPy BGR).
        img_size: Resolusi inferensi model.

    Returns:
        Dictionary hasil prediksi berisi label teratas, confidence, dan sebaran probabilitas.
    """
    img_pil, frame_bgr = _to_pil_and_bgr(img_pil_or_bgr)
    results = model.predict(source=img_pil, verbose=False, device="cpu", imgsz=img_size)
    res = results[0]
    names = res.names
    raw_probs = res.probs

    probs: dict[str, float] = {}
    for idx, name in names.items():
        probs[name.upper()] = float(raw_probs.data[idx])

    # Pastikan seluruh kelas standar terdefinisi
    for cls_name in CLASSES:
        probs.setdefault(cls_name, 0.0)

    top_label = max(probs, key=probs.get)  # type: ignore[arg-type]
    top_conf = probs[top_label]

    annotated_bgr = draw_detection_overlay(frame_bgr, {"label": top_label, "confidence": top_conf})
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return {
        "label": top_label,
        "confidence": top_conf,
        "probs": probs,
        "mode": "single",
        "frame_bgr": frame_bgr,
        "annotated_img": annotated_rgb,
    }


def predict_ensemble(
    model: YOLO,
    img_pil_or_bgr: Image.Image | np.ndarray,
    enable_ensemble: bool | None = None,
    roi_top_ratio: float = FLOOR_ROI_TOP_RATIO,
    img_size: int = IMG_SIZE,
) -> dict[str, Any]:
    """Menjalankan evaluasi Test-Time Spatial Ensemble (Global + Floor Crop + Flip TTA).

    Jika ensemble dinonaktifkan (enable_ensemble=False), otomatis beralih ke predict_single.

    Args:
        model: Instance model YOLOv8.
        img_pil_or_bgr: Citra masukan (PIL Image atau NumPy BGR).
        enable_ensemble: Opsi mengaktifkan ensemble. Jika None, mengikuti ENSEMBLE_ENABLED.
        roi_top_ratio: Rasio batas atas pemotongan fokus lantai.
        img_size: Resolusi inferensi model.

    Returns:
        Dictionary komprehensif berisi label konsensus, confidence, probabilitas gabungan,
        serta rincian skor tiap perspektif.
    """
    is_ensemble = ENSEMBLE_ENABLED if enable_ensemble is None else enable_ensemble

    if not is_ensemble:
        return predict_single(model, img_pil_or_bgr, img_size=img_size)

    img_pil, frame_bgr = _to_pil_and_bgr(img_pil_or_bgr)
    w, h = img_pil.size

    # 1. Perspektif Fokus Lantai (Crop area bawah)
    crop_top = max(1, min(h - 1, int(h * roi_top_ratio)))
    floor_crop = img_pil.crop((0, crop_top, w, h))

    # 2. Perspektif Flip Horizontal (Test-Time Augmentation)
    flip_img = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    # Forward pass batch 3 citra sekaligus dalam 1 panggilan (hemat overhead komputasi)
    try:
        batch_results = model.predict(
            source=[img_pil, floor_crop, flip_img],
            verbose=False,
            device="cpu",
            imgsz=img_size,
        )
        names = batch_results[0].names
        p_global = {names[i].upper(): float(batch_results[0].probs.data[i]) for i in range(len(names))}
        p_crop = {names[i].upper(): float(batch_results[1].probs.data[i]) for i in range(len(names))}
        p_flip = {names[i].upper(): float(batch_results[2].probs.data[i]) for i in range(len(names))}
    except Exception:
        # Fallback aman jika terjadi kendala batch
        fallback = predict_single(model, img_pil, img_size=img_size)
        return fallback

    # Soft voting ensemble terbobot
    p_ens: dict[str, float] = {}
    for c in CLASSES:
        val = (
            ENSEMBLE_WEIGHT_GLOBAL * p_global.get(c, 0.0)
            + ENSEMBLE_WEIGHT_CROP * p_crop.get(c, 0.0)
            + ENSEMBLE_WEIGHT_FLIP * p_flip.get(c, 0.0)
        )
        p_ens[c] = float(val)

    # Normalisasi total probabilitas menjadi tepat 1.0
    tot = sum(p_ens.values()) or 1.0
    p_ens = {c: val / tot for c, val in p_ens.items()}

    top_label = max(p_ens, key=p_ens.get)  # type: ignore[arg-type]
    top_conf = p_ens[top_label]

    annotated_bgr = draw_detection_overlay(frame_bgr, {"label": top_label, "confidence": top_conf}, roi_top_ratio=roi_top_ratio)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return {
        "label": top_label,
        "confidence": top_conf,
        "probs": p_ens,
        "mode": "ensemble",
        "frame_bgr": frame_bgr,
        "annotated_img": annotated_rgb,
        "details": {
            "global": p_global,
            "crop": p_crop,
            "flip": p_flip,
            "floor_crop": p_crop,
            "flip_tta": p_flip,
            "weights": {
                "global": ENSEMBLE_WEIGHT_GLOBAL,
                "crop": ENSEMBLE_WEIGHT_CROP,
                "flip": ENSEMBLE_WEIGHT_FLIP,
            },
        },
    }


def draw_detection_overlay(
    frame_bgr: np.ndarray,
    pred: dict[str, Any],
    roi_top_ratio: float = FLOOR_ROI_TOP_RATIO,
) -> np.ndarray:
    """Menggambar batas drop zone pemantauan lantai dan banner status inferensi."""
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    # Gambar batas Drop Zone pemantauan lantai
    roi_y = int(h * roi_top_ratio)
    cv2.rectangle(annotated, (8, roi_y), (w - 8, h - 8), (255, 190, 0), 2)
    cv2.putText(
        annotated,
        "ZONA PEMANTAUAN LANTAI / RAK SANDAL",
        (15, roi_y - 10 if roi_y > 25 else roi_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 190, 0),
        2,
        cv2.LINE_AA,
    )

    label = pred.get("label", "KOSONG")
    conf = pred.get("confidence", 0.0)

    if label == "BERANTAKAN":
        banner_color = (0, 0, 220)  # Red
        status_text = f"PERINGATAN! BERANTAKAN ({conf*100:.1f}%)"
        sub_text = "Sandal/sepatu berserakan di depan pintu masuk lab!"
    elif label == "RAPI":
        banner_color = (0, 200, 0)  # Green
        status_text = f"STATUS: RAPI ({conf*100:.1f}%)"
        sub_text = "Sandal/sepatu tertata rapi di rak / teratur berdampingan"
    else:
        banner_color = (220, 140, 20)  # Blue-cyan (BGR)
        status_text = f"STATUS: KOSONG / STANDBY ({conf*100:.1f}%)"
        sub_text = "Area lantai bersih (tidak ada sandal atau sepatu)"

    banner_height = max(75, int(h * 0.12))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    font_scale_main = max(0.65, w / 900.0)
    font_scale_sub = max(0.5, w / 1200.0)

    cv2.putText(
        annotated,
        status_text,
        (15, int(banner_height * 0.45)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale_main,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        sub_text,
        (15, int(banner_height * 0.82)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale_sub,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated
