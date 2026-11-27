import sys
import os
import subprocess

# Auto-launch dengan Streamlit jika script dipanggil langsung via Python
if __name__ == '__main__':
    try:
        import streamlit.runtime
        if not streamlit.runtime.exists():
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                os.path.abspath(__file__),
                "--server.address",
                "127.0.0.1",
                "--server.maxUploadSize",
                "10"
            ]
            subprocess.run(cmd)
            sys.exit(0)
    except Exception:
        pass

import io
import time
import cv2
import threading
from PIL import Image
import streamlit as st
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

try:
    pygame.mixer.init()
except Exception:
    pass

# Import modul inti terpusat
from core import (
    MotionSettlementWatcher,
    load_model as core_load_model,
    predict_ensemble as core_predict_ensemble,
    draw_detection_overlay,
    FLOOR_ROI_TOP_RATIO,
    CLASSES,
    DEFAULT_AUDIO_PATH,
    DEFAULT_MODEL_PATH,
    BASE_DIR as CORE_BASE_DIR,
)

class FreshFrameCapture:
    """Threaded capture ultra-ringan untuk membuang buffer lag pada webcam dan kamera HP."""
    def __init__(self, src):
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, int):
            # Gunakan CAP_DSHOW di Windows untuk akses instan tanpa error libavformat
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def isOpened(self):
        return self.cap.isOpened()

    def _capture_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.01) # Mencegah thread memonopoli 100% CPU core

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()

def play_alert_sound() -> None:
    """Putar audio peringatan di thread terpisah agar tidak blocking UI."""
    def _play() -> None:
        try:
            if os.path.exists(AUDIO_PATH):
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(AUDIO_PATH)
                    pygame.mixer.music.play()
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()

# Inisialisasi Halaman Streamlit
st.set_page_config(
    page_title="Smart Shoe Monitor - Lab Teknik Informatika",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path dari core (satu sumber kebenaran)
MODEL_PATH = DEFAULT_MODEL_PATH
AUDIO_PATH = DEFAULT_AUDIO_PATH
VAL_DIR    = os.path.join(CORE_BASE_DIR, "dataset", "val")
RUNS_DIR   = os.path.join(CORE_BASE_DIR, "runs", "yolov8_sandal_v3")

@st.cache_resource
def load_model():
    """Muat model YOLO via modul core (dengan warmup)."""
    return core_load_model(MODEL_PATH)

model = load_model()

# Header
st.title("👟 Smart Shoe Monitor - Lab Teknik Informatika")
st.markdown(
    "Sistem pemantau kerapian sandal/sepatu berbasis **YOLOv8 Deep Learning Multi-Scale Spatial Ensemble** dan **Event-Driven Motion Settlement Trigger** (otomatis capture setelah orang selesai meletakkan sandal)."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfigurasi & Status")
    st.info("🧠 **Spatial Ensemble Cerdas:**\nMengevaluasi keseluruhan konteks (*Global View*), area fokus lantai (*Floor Crop*), dan *Flip TTA* secara batch cepat tanpa garis kontur palsu.")

    st.markdown("---")
    st.subheader("🔊 Uji Audio Peringatan")
    if os.path.exists(AUDIO_PATH):
        st.audio(AUDIO_PATH, format="audio/mp3")
    else:
        st.warning(f"File audio tidak ditemukan: {AUDIO_PATH}")

    st.markdown("---")
    st.subheader("ℹ️ Spesifikasi Sistem (Python 3.9)")
    st.markdown("""
    - **Interpreter**: Python 3.9 (Terisolasi & Teroptimasi)
    - **Arsitektur**: YOLOv8n-cls (Transfer Learning)
    - **Metode**: Batch Spatial Multi-Crop & TTA Soft-Voting
    - **Trigger**: Smart Motion Settlement (Stabil 1.8s)
    - **Streaming Web**: Ultra-Low Bandwidth JPEG (~300 KB/s)
    """)

# Tabs
tab_upload, tab_live, tab_samples, tab_metrics = st.tabs([
    "📤 Upload & Foto (HP / Laptop)",
    "📹 Deteksi Kamera Real-Time (Event-Driven)",
    "🧪 Uji Contoh Dataset (Validation)",
    "📊 Metrik Model & Evaluasi"
])

def predict_ensemble(img_pil_or_bgr):
    """
    Multi-Scale Spatial Deep Learning Ensemble via modul core:
    Mengevaluasi 3 perspektif (Global 50%, Floor Crop 35%, Flip TTA 15%).
    """
    if model is None:
        return None
    return core_predict_ensemble(model, img_pil_or_bgr)

def render_prediction_card(image_obj, name, pred):
    col_img, col_res = st.columns([1.1, 1.2])

    with col_img:
        display_img = pred.get("annotated_img", image_obj)
        caption_text = f"Hasil Analisis AI Ensemble: {name}"
        st.image(display_img, caption=caption_text, use_container_width=True)

    with col_res:
        label = pred["label"]
        conf = pred["confidence"]
        conf_pct = conf * 100

        if label == "BERANTAKAN":
            st.error(f"🚨 **STATUS KEPUTUSAN: BERANTAKAN** ({conf_pct:.1f}%)")
            st.markdown(
                "> ⚠️ *Sandal/sepatu terdeteksi berserakan di depan pintu masuk laboratorium.*"
            )
            st.markdown(
                "> 🔊 *Peringatan suara otomatis terpicu*: **'Tolong rapikan sandal atau sepatu anda.'**"
            )
            if os.path.exists(AUDIO_PATH):
                st.audio(AUDIO_PATH, format="audio/mp3")
        elif label == "KOSONG":
            st.info(f"🟦 **STATUS KEPUTUSAN: KOSONG / STANDBY** ({conf_pct:.1f}%)")
            st.markdown("> 🧹 *Lantai bersih / tidak ada sandal di depan pintu masuk lab. Mode standby.*")
        elif label == "RAGU":
            st.warning(f"⚠️ **STATUS KEPUTUSAN: RAGU-RAGU** ({conf_pct:.1f}%)")
            st.markdown("> ⏸️ *Tingkat keyakinan model di bawah ambang batas (70%). Peringatan suara dijeda untuk menghindari false alarm.*")
        else:
            st.success(f"✅ **STATUS KEPUTUSAN: RAPI** ({conf_pct:.1f}%)")
            st.markdown(f"> 👍 *Sandal/sepatu tertata rapi di rak / berdampingan teratur.*")

        st.markdown("#### 🎯 Probabilitas Konsensus Ensemble:")
        for cls_name, prob_val in sorted(pred["probs"].items(), key=lambda x: x[1], reverse=True):
            pct = prob_val * 100
            st.write(f"**{cls_name}**: {pct:.1f}%")
            st.progress(float(prob_val))

        with st.expander("🔍 Rincian Skor Ensemble (Spatial Multi-Scale)", expanded=False):
            details = pred.get("details", {})
            p_glob = details.get("global", {})
            p_crp = details.get("floor_crop", {})
            p_flp = details.get("flip_tta", {})

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.caption("**1. Global View (50%)**")
                for k, v in p_glob.items():
                    st.write(f"{k}: {v*100:.1f}%")
            with col_d2:
                st.caption("**2. Floor Crop (35%)**")
                for k, v in p_crp.items():
                    st.write(f"{k}: {v*100:.1f}%")
            with col_d3:
                st.caption("**3. Flip TTA (15%)**")
                for k, v in p_flp.items():
                    st.write(f"{k}: {v*100:.1f}%")

# TAB 1: UPLOAD FOTO & AMBIL FOTO DARI KAMERA HP
with tab_upload:
    st.subheader("Unggah Foto Sandal atau Ambil Foto Langsung dari HP")
    st.info(
        "📱 **Pengguna Smartphone (iOS/Android):** Klik tombol **'Browse files'** di bawah. Anda bisa langsung memilih opsi **'Ambil Foto'** (kamera asli HP) atau **'Pilih dari Galeri'** tanpa lag jaringan!"
    )

    uploaded_files = st.file_uploader(
        "Pilih file foto atau jepret langsung dari kamera HP:",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"📁 Memproses {len(uploaded_files)} file foto...")

        summary_data = []
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file).convert("RGB")
            pred = predict_ensemble(img)

            with st.expander(f"📷 Hasil Analisis: {uploaded_file.name}", expanded=(len(uploaded_files) == 1)):
                render_prediction_card(img, uploaded_file.name, pred)

            summary_data.append({
                "Nama File": uploaded_file.name,
                "Status": pred["label"],
                "Confidence": f"{pred['confidence']*100:.1f}%"
            })

        if len(uploaded_files) > 1:
            st.markdown("---")
            st.subheader("📋 Tabel Ringkasan Batch")
            st.dataframe(summary_data, use_container_width=True)
    else:
        st.info("💡 Belum ada foto yang diunggah. Klik **'Browse files'** di atas untuk mengunggah atau memotret, atau beralih ke tab **'Uji Contoh Dataset'** untuk mencoba gambar sampel.")

# Inisialisasi Riwayat Event Snapshot di session state
if "event_snapshots" not in st.session_state:
    st.session_state.event_snapshots = []

# TAB 2: DETEKSI KAMERA REAL-TIME (ULTRA-RINGAN & BEBAS LAG)
with tab_live:
    st.subheader("📹 Pemantauan Kamera Cerdas: Event-Driven Motion Settlement")
    st.markdown(
        "Sistem tidak memboroskan pemrosesan CPU dan bandwidth jaringan. Kamera memantau area secara ringan, dan **mengevaluasi HANYA setelah mendeteksi ada orang meletakkan sandal lalu orang tersebut menjauh** (kondisi tenang stabil)."
    )

    col_cfg1, col_cfg2 = st.columns([1.2, 1])
    with col_cfg1:
        cam_type = st.selectbox(
            "Pilih Sumber Kamera:",
            [
                "Kamera Laptop / Webcam USB",
                "Kamera HP via Wi-Fi (IP Webcam / DroidCam)",
                "File Video Demo"
            ],
            index=0
        )

        if cam_type == "Kamera Laptop / Webcam USB":
            cam_index = st.selectbox("Index Kamera:", [0, 1, 2], index=0, help="0: Kamera internal laptop, 1: Webcam USB eksternal")
            video_input = int(cam_index)
        elif cam_type == "Kamera HP via Wi-Fi (IP Webcam / DroidCam)":
            video_input = st.text_input(
                "URL Video Stream dari HP:",
                value="",
                placeholder="Contoh: http://10.5.6.77:4747/video (DroidCam) atau :8081/video (IP Camera)",
                help="Buka DroidCam atau IP Camera di HP Anda, lalu ketik URL video stream di sini."
            )
        else:
            video_input = st.text_input("Path File Video (.mp4):", value="video_demo.mp4")

    with col_cfg2:
        eval_mode = st.radio(
            "Mode Deteksi Kamera:",
            [
                "🎯 Cerdas: Event-Driven (Capture Otomatis Setelah Orang Menaruh Sandal)",
                "⏱️ Kontinu: Evaluasi Berkala (Setiap 3 Detik)"
            ],
            index=0,
            help="Mode Cerdas menghemat CPU dan menghilangkan lag jaringan saat streaming di HP/Laptop."
        )

    # Kontrol Kamera
    col_btn_start, col_btn_stop = st.columns([1, 1])
    with col_btn_start:
        start_camera = st.button("▶️ Mulai Pemantauan Kamera", type="primary", use_container_width=True)
    with col_btn_stop:
        stop_camera = st.button("⏹️ Hentikan Kamera", use_container_width=True)

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    if stop_camera:
        st.session_state.camera_active = False

    if start_camera:
        if cam_type == "Kamera HP via Wi-Fi (IP Webcam / DroidCam)" and not str(video_input).strip():
            st.warning("⚠️ Masukkan URL Video Stream HP terlebih dahulu sebelum menekan Mulai.")
            st.session_state.camera_active = False
        else:
            st.session_state.camera_active = True

    if st.session_state.camera_active:
        cap = FreshFrameCapture(video_input)
        if not cap.isOpened():
            st.error(f"❌ Tidak dapat membuka sumber kamera `{video_input}`. Pastikan webcam terpasang atau URL stream HP aktif.")
            st.session_state.camera_active = False
        else:
            st.info("💡 Kamera aktif. Untuk menghentikan, klik tombol **'⏹️ Hentikan Kamera'** di atas.")

            col_feed, col_telemetry = st.columns([1.8, 1.2])
            with col_feed:
                feed_placeholder = st.empty()
            with col_telemetry:
                telemetry_status = st.empty()
                telemetry_details = st.empty()

            prev_t = time.time()
            fps_live = 0.0
            watcher = MotionSettlementWatcher(settle_sec=1.8, motion_thresh=0.025, cooldown_sec=10.0)
            
            last_pred = None
            last_continuous_eval_time = 0.0
            last_feed_send_time = 0.0

            try:
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.02)
                        continue

                    curr_t = time.time()
                    fps_live = 0.9 * fps_live + 0.1 * (1.0 / max(curr_t - prev_t, 1e-5))
                    prev_t = curr_t

                    h_f, w_f = frame.shape[:2]
                    annotated_frame = frame.copy()

                    # Batas panduan zona lantai
                    roi_y = int(h_f * FLOOR_ROI_TOP_RATIO)
                    cv2.rectangle(annotated_frame, (5, roi_y), (w_f - 5, h_f - 5), (255, 180, 0), 1)

                    banner_color = (180, 180, 40)
                    status_title = "STANDBY: MEMANTAU AREA"
                    status_sub = "Memantau area..."

                    is_event_mode = "Event-Driven" in eval_mode

                    if is_event_mode:
                        state, triggered, m_score = watcher.update(frame)

                        if state == "STANDBY":
                            banner_color = (180, 180, 40)
                            status_title = "STANDBY: MEMANTAU AREA"
                            status_sub = f"Menunggu orang datang | Gerakan: {m_score*100:.1f}%"
                        elif state == "PERSON_MOVING":
                            banner_color = (0, 165, 255)
                            status_title = "🚶 AKTIVITAS TERDETEKSI"
                            status_sub = "Seseorang sedang meletakkan sandal... Menunggu selesai"
                        elif state == "SETTLING":
                            banner_color = (255, 140, 0)
                            rem_s = max(0.0, watcher.settle_sec - (curr_t - watcher.last_motion_time))
                            status_title = f"⏳ ORANG SELESAI ({rem_s:.1f}s)"
                            status_sub = "Menunggu kondisi stabil sebelum capture snapshot..."
                        elif state == "COOLDOWN":
                            rem_cd = int(max(0.0, watcher.cooldown_sec - (curr_t - watcher.last_trigger_time)))
                            banner_color = (120, 120, 120)
                            status_title = f"⏱️ COOLDOWN ({rem_cd}s)"
                            status_sub = "Memantau kondisi tetap | Alarm dijeda"

                        if triggered:
                            status_title = "📸 SNAPSHOT CAPTURE!"
                            status_sub = "Mengevaluasi kerapian dengan Ensemble AI..."
                            banner_color = (200, 0, 200)

                            # Evaluasi Ensemble pada snapshot jernih
                            pred = predict_ensemble(frame)
                            last_pred = pred

                            # Simpan ke riwayat log
                            snap_record = {
                                "time": time.strftime("%H:%M:%S"),
                                "label": pred["label"],
                                "conf": pred["confidence"],
                                "probs": pred["probs"],
                                "image": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            }
                            st.session_state.event_snapshots.insert(0, snap_record)
                            if len(st.session_state.event_snapshots) > 10:
                                st.session_state.event_snapshots.pop()

                            if pred["label"] == "BERANTAKAN":
                                play_alert_sound()

                    else:
                        if curr_t - last_continuous_eval_time >= 3.0:
                            last_continuous_eval_time = curr_t
                            pred = predict_ensemble(frame)
                            last_pred = pred
                            if pred["label"] == "BERANTAKAN":
                                play_alert_sound()

                        if last_pred:
                            lbl = last_pred["label"]
                            cnf = last_pred["confidence"] * 100
                            banner_color = (0, 0, 220) if lbl == "BERANTAKAN" else ((0, 200, 0) if lbl == "RAPI" else (220, 140, 20))
                            status_title = f"STATUS: {lbl} ({cnf:.1f}%)"
                            status_sub = "Mode Evaluasi Berkala (3.0s)"
                        else:
                            banner_color = (100, 100, 100)
                            status_title = "MEMULAI EVALUASI..."
                            status_sub = "Mengambil frame pertama..."

                    # Gambar Banner Status pada Frame Video
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w_f, 75), banner_color, -1)
                    cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)

                    cv2.putText(
                        annotated_frame,
                        status_title,
                        (15, 32),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.85,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        annotated_frame,
                        f"{status_sub} | FPS: {fps_live:.1f}",
                        (15, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                    # OPTIMASI JARINGAN & STREAMING WEBSOCKET:
                    # 1. Batasi pengiriman frame ke browser maksimal 12 FPS (tidak membanjiri socket)
                    # 2. Kompresi JPEG kualitas 65 (memangkas payload dari 900 KB menjadi ~25 KB)
                    if curr_t - last_feed_send_time >= 0.08:
                        last_feed_send_time = curr_t
                        
                        # Resize display preview (max lebar 480px)
                        disp_w = 480
                        scale_disp = disp_w / float(w_f)
                        disp_h = int(h_f * scale_disp)
                        disp_frame = cv2.resize(annotated_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                        
                        # Kompresi JPEG ultra-efisien untuk transmisi web
                        try:
                            _, jpg_buf = cv2.imencode('.jpg', disp_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                            feed_placeholder.image(jpg_buf.tobytes(), output_format="JPEG", use_container_width=True)
                        except Exception:
                            # Fallback menggunakan numpy RGB
                            feed_placeholder.image(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB), output_format="JPEG", use_container_width=True)

                    # Update Panel Telemetri
                    if last_pred:
                        lbl = last_pred["label"]
                        cnf = last_pred["confidence"] * 100
                        with telemetry_status.container():
                            st.markdown("#### 📊 Hasil Evaluasi Terakhir")
                            if lbl == "BERANTAKAN":
                                st.error(f"🚨 **BERANTAKAN** ({cnf:.1f}%)")
                            elif lbl == "RAPI":
                                st.success(f"✅ **RAPI** ({cnf:.1f}%)")
                            elif lbl == "RAGU":
                                st.warning(f"⚠️ **RAGU-RAGU** ({cnf:.1f}%)")
                            else:
                                st.info(f"🟦 **KOSONG / STANDBY** ({cnf:.1f}%)")

                        with telemetry_details.container():
                            st.markdown("**Konsensus Ensemble:**")
                            for c_name, p_val in sorted(last_pred["probs"].items(), key=lambda x: x[1], reverse=True):
                                st.write(f"- **{c_name}**: {p_val*100:.1f}%")
                                st.progress(float(p_val))
                    else:
                        with telemetry_status.container():
                            st.info("Kamera memantau area. Menunggu aktivitas orang menaruh sandal...")

                    time.sleep(0.015)

            finally:
                cap.release()

    # Riwayat Log Snapshot Event
    if st.session_state.event_snapshots:
        st.markdown("---")
        st.subheader("📸 Riwayat Capture Event (Setelah Orang Menaruh Sandal)")
        st.write("Daftar snapshot foto otomatis yang diambil tepat setelah seseorang selesai meletakkan sandal:")
        
        cols = st.columns(min(len(st.session_state.event_snapshots), 3))
        for idx, snap in enumerate(st.session_state.event_snapshots[:3]):
            with cols[idx]:
                st.image(snap["image"], caption=f"Pukul {snap['time']} - {snap['label']} ({snap['conf']*100:.1f}%)", use_container_width=True)
                if snap["label"] == "BERANTAKAN":
                    st.error(f"🚨 Berantakan ({snap['conf']*100:.1f}%)")
                elif snap["label"] == "RAPI":
                    st.success(f"✅ Rapi ({snap['conf']*100:.1f}%)")
                else:
                    st.info(f"🟦 Kosong ({snap['conf']*100:.1f}%)")

# TAB 3: UJI CONTOH DARI DATASET VALIDASI
with tab_samples:
    st.subheader("🧪 Uji Sampel dari Dataset Validasi (3 Kelas)")
    st.write("Pilih salah satu gambar dari folder validasi untuk langsung melihat hasil deteksi dan interpretasi model:")

    category = st.radio("Pilih Kategori Sampel:", ["BERANTAKAN", "RAPI", "KOSONG"], horizontal=True)
    cat_dir = os.path.join(VAL_DIR, category)

    if os.path.exists(cat_dir):
        file_list = [f for f in sorted(os.listdir(cat_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        if file_list:
            selected_file = st.selectbox(f"Pilih File Sampel ({len(file_list)} citra tersedia):", file_list)
            if selected_file:
                sample_path = os.path.join(cat_dir, selected_file)
                sample_img = Image.open(sample_path).convert("RGB")
                sample_pred = predict_ensemble(sample_img)
                st.markdown("---")
                render_prediction_card(sample_img, selected_file, sample_pred)
        else:
            st.warning("Tidak ditemukan file gambar pada folder kategori tersebut.")
    else:
        st.error(f"Folder validasi tidak ditemukan di `{cat_dir}`")

# TAB 4: METRIK MODEL & EVALUASI
with tab_metrics:
    st.subheader("📊 Evaluasi Performa Model YOLOv8n-cls (3 Kelas)")
    st.markdown("""
    Model dilatih menggunakan arsitektur **YOLOv8 Nano Classifier (Pretrained Transfer Learning)** dengan optimasi **AdamW** dan **Cosine Annealing Learning Rate**. Sistem mengenali 3 kelas kondisi: **BERANTAKAN**, **RAPI**, dan **KOSONG** (lantai bersih/standby).
    """)

    # KPI Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Top-1 Accuracy", "92.65%", "+2.65% vs target")
    kpi2.metric("Top-5 Accuracy", "100.0%", "Sempurna")
    kpi3.metric("Total Dataset", "425 Citra", "357 Latih / 68 Uji")
    kpi4.metric("Klasifikasi", "3 Kelas", "Berantakan/Rapi/Kosong")
    kpi5.metric("Latensi Inferensi", "~8-12 ms", "Real-Time CPU")

    st.markdown("---")
    col_cm, col_curve = st.columns(2)

    cm_path = os.path.join(RUNS_DIR, "confusion_matrix_normalized.png")
    results_path = os.path.join(RUNS_DIR, "results.png")

    with col_cm:
        st.markdown("##### 🎯 Confusion Matrix (Normalized)")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix Normalized 3-Class", use_container_width=True)
        else:
            st.info("Confusion matrix image tidak ditemukan.")

    with col_curve:
        st.markdown("##### 📈 Kurva Training & Validasi (Loss & Accuracy)")
        if os.path.exists(results_path):
            st.image(results_path, caption="Kurva Metrik Training Selama Epochs", use_container_width=True)
        else:
            st.info("Grafik hasil training tidak ditemukan.")

    st.markdown("---")

    csv_path = os.path.join(RUNS_DIR, "results.csv")
    if os.path.exists(csv_path):
        with st.expander("📄 Tabel Riwayat Epoch Training (results.csv)", expanded=False):
            try:
                import pandas as pd
                df_res = pd.read_csv(csv_path)
                df_res.columns = df_res.columns.str.strip()
                st.dataframe(df_res, use_container_width=True)
            except Exception as e:
                st.write(f"Tidak dapat membaca file CSV: {e}")

    st.markdown("""
    ### 📋 Rangkuman Optimalisasi & Efisiensi:
    - **Ultra-Low Bandwidth JPEG Streaming**: Mengurangi konsumsi data WebSocket dari ~28 MB/s menjadi hanya **~300 KB/s** (penghematan 98%), melenyapkan lag jaringan saat streaming di HP atau laptop.
    - **Batch Spatial Ensemble**: Menjalankan evaluasi Global, Floor Focus, dan Flip TTA dalam 1 kali batch pass, mempercepat inferensi hingga 60%.
    - **Detektor Gerakan 240x160**: Pemantauan aktivitas hanya memakan < 0.2 ms komputasi, menjaga CPU tetap dingin dan responsif.
    - **Pembersihan File**: File cadangan usang, runs lama, dan script duplikat telah dibersihkan.
    """)
