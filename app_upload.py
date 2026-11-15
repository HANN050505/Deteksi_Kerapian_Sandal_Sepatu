import sys
import os

# Auto-launch with streamlit run if executed directly with python
try:
    import streamlit.runtime
    if not streamlit.runtime.exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
except Exception:
    pass

import io
import time
import cv2
import threading
from PIL import Image
import streamlit as st
import numpy as np
from ultralytics import YOLO

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
import pygame

try:
    pygame.mixer.init()
except Exception:
    pass

class FreshFrameCapture:
    """Threaded capture to eliminate OpenCV buffer lag on IP cameras & webcams."""
    def __init__(self, src):
        if isinstance(src, str) and src.isdigit():
            src = int(src)
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
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()

def play_alert_sound():
    def _play():
        try:
            if os.path.exists(AUDIO_PATH):
                pygame.mixer.music.load(AUDIO_PATH)
                pygame.mixer.music.play()
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()

# Set Page Config
st.set_page_config(
    page_title="Smart Shoe Monitor - Upload & Uji Foto",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
AUDIO_PATH = os.path.join(BASE_DIR, "audio", "peringatan.mp3")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "val")
RUNS_DIR = (
    os.path.join(BASE_DIR, "runs", "yolov8_sandal_v3")
    if os.path.exists(os.path.join(BASE_DIR, "runs", "yolov8_sandal_v3"))
    else (
        os.path.join(BASE_DIR, "runs", "yolov8_sandal_v2")
        if os.path.exists(os.path.join(BASE_DIR, "runs", "yolov8_sandal_v2"))
        else os.path.join(BASE_DIR, "runs", "yolov8_sandal")
    )
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan di `{MODEL_PATH}`")
        return None
    return YOLO(MODEL_PATH)

model = load_model()

# Header
st.title("👟 Smart Shoe Monitor - Lab Teknik Informatika")
st.markdown(
    "Aplikasi pengujian model klasifikasi **YOLOv8 Nano (3-Kelas)** untuk memantau kerapian alas kaki di depan pintu masuk **Laboratorium Komputer & Teknik Informatika**: mendeteksi kondisi **RAPI**, **BERANTAKAN**, atau **KOSONG** (lantai bersih/standby)."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan & Info")
    st.info("🎯 **Mode Keputusan Pasti:**\nSistem langsung menentukan status **RAPI** atau **BERANTAKAN** berdasarkan model YOLO + deteksi kotak fisik pasangan sandal (tanpa status ragu-ragu).")

    st.markdown("---")
    st.subheader("🔊 Uji Audio Peringatan")
    if os.path.exists(AUDIO_PATH):
        st.audio(AUDIO_PATH, format="audio/mp3")
    else:
        st.warning(f"File audio tidak ditemukan: {AUDIO_PATH}")

    st.markdown("---")
    st.subheader("ℹ️ Spesifikasi Model")
    st.markdown("""
    - **Model AI**: YOLOv8n-cls + Deteksi Kotak Pasangan Sandal
    - **Resolusi**: 224 x 224 px
    - **Status Utama**: `RAPI` & `BERANTAKAN`
    - **Logika Pasangan**: Kotak Hijau = Pasangan Rapi, Kotak Merah = Tercecer
    """)

# Tabs
tab_upload, tab_live, tab_samples, tab_metrics = st.tabs([
    "📤 Upload & Foto (HP / Laptop)",
    "📹 Deteksi Kamera Real-Time",
    "🧪 Uji Contoh Dataset (Validation)",
    "📊 Metrik Model & Evaluasi"
])

def analyze_and_draw_shoe_pairs(image_bgr):
    """
    Mendeteksi kontur fisik sandal/sepatu, menganalisis pasangan (pairing),
    dan menggambar kotak bounding box:
    - KOTAK HIJAU: Pasangan Rapi (berdampingan & sejajar)
    - KOTAK MERAH: Tidak Berpasangan / Tercecer (meningkatkan skor berantakan)
    """
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
            
    # Gambar Kotak HIJAU untuk sandal berpasangan
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

    # Gambar Kotak MERAH untuk sandal yang tercecer / tidak berpasangan
    for u in unpaired:
        b = u['box']
        cv2.rectangle(annotated, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 0, 255), 2)
        cv2.putText(annotated, "TIDAK BERPASANGAN", (b[0], max(b[1] - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    return annotated, len(candidates), len(pairs), len(unpaired)

def predict_pil_image(img_pil):
    if model is None:
        return None
    results = model.predict(source=img_pil, verbose=False, device='cpu', imgsz=224)
    result = results[0]
    probs = result.probs
    names = result.names
    
    top1_idx = probs.top1
    raw_label = names[top1_idx].upper()
    top1_conf = float(probs.top1conf)
    
    prob_dict = {}
    for idx, name in names.items():
        prob_dict[name.upper()] = float(probs.data[idx])
        
    # Analisis Kotak Bounding Box Pasangan Sandal
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    annotated_bgr, total_sandals, num_pairs, num_unpaired = analyze_and_draw_shoe_pairs(img_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # Logika Keputusan Pasti: Hanya RAPI atau BERANTAKAN (atau KOSONG)
    if total_sandals == 0 and raw_label == "KOSONG":
        final_label = "KOSONG"
    elif num_unpaired > 0 or raw_label == "BERANTAKAN":
        final_label = "BERANTAKAN"
    else:
        final_label = "RAPI"
        
    return {
        "label": final_label,
        "raw_label": raw_label,
        "confidence": top1_conf,
        "probs": prob_dict,
        "annotated_img": annotated_rgb,
        "total_sandals": total_sandals,
        "num_pairs": num_pairs,
        "num_unpaired": num_unpaired
    }

def render_prediction_card(image_obj, name, pred):
    col_img, col_res = st.columns([1, 1.2])
    
    with col_img:
        display_img = pred.get("annotated_img", image_obj)
        st.image(display_img, caption=f"{name} (Deteksi Kotak Pasangan Sandal)", width="stretch")
        
    with col_res:
        label = pred["label"]
        conf = pred["confidence"]
        conf_pct = conf * 100
        total_s = pred.get("total_sandals", 0)
        num_p = pred.get("num_pairs", 0)
        num_u = pred.get("num_unpaired", 0)
        
        st.markdown("##### 📦 Deteksi Bounding Box Fisik Sandal:")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Terdeteksi", f"{total_s}")
        c2.metric("🟢 Pasang Rapi", f"{num_p}")
        c3.metric("🔴 Tercecer", f"{num_u}")
        
        if label == "BERANTAKAN":
            st.error(f"🚨 **STATUS: BERANTAKAN**")
            if num_u > 0:
                st.markdown(f"> ⚠️ *Terdeteksi **{num_u} sandal tidak berpasangan / tercecer** sehingga menambah akurasi berantakan.*")
            st.markdown(
                "> 🔊 *Peringatan otomatis terpicu*: **'Tolong rapikan sandal atau sepatu anda.'**"
            )
            if os.path.exists(AUDIO_PATH):
                st.audio(AUDIO_PATH, format="audio/mp3")
        elif label == "KOSONG":
            st.info(f"🟦 **STATUS: KOSONG / STANDBY**")
            st.markdown("> 🧹 *Area lantai bersih, tidak ada sandal atau sepatu yang terdeteksi.*")
        else: # RAPI
            st.success(f"✅ **STATUS: RAPI**")
            st.markdown(f"> 👍 *Kondisi tertata rapi (**{num_p} pasang** tersusun rapi berpasangan).*")
            
        st.markdown("#### Detail Probabilitas Model:")
        for cls_name, prob_val in sorted(pred["probs"].items(), key=lambda x: x[1], reverse=True):
            pct = prob_val * 100
            st.write(f"**{cls_name}**: {pct:.1f}%")
            st.progress(float(prob_val))
            st.progress(float(prob_val))

# TAB 1: UPLOAD FOTO & AMBIL FOTO DARI KAMERA HP
with tab_upload:
    st.subheader("Unggah Foto Sandal atau Ambil Foto Langsung dari HP")
    st.info(
        "📱 **Pengguna Smartphone (iOS/Android):** Klik tombol **'Browse files'** di bawah. Anda bisa langsung memilih opsi **'Ambil Foto'** (kamera asli HP) atau **'Pilih dari Galeri'** tanpa perlu izin WebRTC / HTTPS browser yang rumit!"
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
            pred = predict_pil_image(img)
            
            with st.expander(f"📷 Hasil Analisis: {uploaded_file.name}", expanded=(len(uploaded_files) == 1)):
                render_prediction_card(img, uploaded_file.name, pred)
                
            status_desc = pred["label"]
            summary_data.append({
                "Nama File": uploaded_file.name,
                "Status": pred["label"],
                "Pasang Rapi": pred.get("num_pairs", 0),
                "Tercecer": pred.get("num_unpaired", 0),
                "Confidence": f"{pred['confidence']*100:.1f}%"
            })
            
        if len(uploaded_files) > 1:
            st.markdown("---")
            st.subheader("📋 Tabel Ringkasan Batch")
            st.dataframe(summary_data, width="stretch")
    else:
        st.info("💡 Belum ada foto yang diunggah. Klik **'Browse files'** di atas untuk mengunggah atau memotret, atau beralih ke tab **'Uji Contoh Dataset'** untuk mencoba gambar sampel.")

# TAB 2: DETEKSI KAMERA REAL-TIME
with tab_live:
    st.subheader("📹 Pemantauan Kamera Video Real-Time")
    st.markdown(
        "Gunakan kamera langsung untuk memantau kerapian sandal di depan pintu masuk **Laboratorium Teknik Informatika**."
    )

    col_src_type, col_src_val = st.columns([1, 1.2])
    with col_src_type:
        cam_type = st.selectbox(
            "Pilih Sumber Kamera:",
            [
                "Kamera Laptop / Webcam USB",
                "Kamera HP via Wi-Fi (IP Webcam / DroidCam)",
                "File Video Demo"
            ],
            index=0
        )

    with col_src_val:
        if cam_type == "Kamera Laptop / Webcam USB":
            cam_index = st.selectbox("Index Kamera:", [0, 1, 2], index=0, help="0: Kamera internal laptop, 1: Webcam USB eksternal")
            video_input = int(cam_index)
        elif cam_type == "Kamera HP via Wi-Fi (IP Webcam / DroidCam)":
            video_input = st.text_input(
                "URL Video Stream dari HP:",
                value="",
                placeholder="Contoh: http://10.5.6.77:4747/video (DroidCam) atau :8081/video (IP Camera)",
                help="Cara pakai: Di iPhone/Android buka aplikasi 'DroidCam' atau 'IP Camera Lite'. Lihat IP yang muncul di layar HP Anda, lalu ketik di sini dengan format: http://[IP_HP]:4747/video"
            )
        else:
            video_input = st.text_input("Path File Video (.mp4):", value="video_demo.mp4")

    # Kontrol Start / Stop Streaming
    col_btn_start, col_btn_stop = st.columns([1, 1])
    with col_btn_start:
        start_camera = st.button("▶️ Mulai Pemantauan Kamera Real-Time", type="primary", use_container_width=True)
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
            st.info("💡 Kamera aktif secara real-time. Untuk menghentikan, klik tombol **'⏹️ Hentikan Kamera'** di atas.")

            col_feed, col_telemetry = st.columns([2, 1])
            with col_feed:
                feed_placeholder = st.empty()
            with col_telemetry:
                telemetry_status = st.empty()
                telemetry_chart = st.empty()

            prev_t = time.time()
            fps_live = 0.0
            messy_start_time = None
            last_alert_time = 0.0
            cooldown_sec = 12.0
            trigger_sec = 2.5

            try:
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.02)
                        continue

                    # Auto-downscale large resolution from phone (1080p -> 640px) to prevent lag
                    if frame.shape[1] > 640:
                        scale = 640.0 / frame.shape[1]
                        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

                    curr_t = time.time()
                    fps_live = 0.9 * fps_live + 0.1 * (1.0 / max(curr_t - prev_t, 1e-5))
                    prev_t = curr_t

                    # Analisis deteksi kotak bounding box fisik sandal & pasangan
                    annotated_frame, tot_s_live, num_p_live, num_u_live = analyze_and_draw_shoe_pairs(frame)

                    # Predict frame with YOLO
                    results = model.predict(source=frame, verbose=False, device='cpu', imgsz=224)
                    result = results[0]
                    probs = result.probs
                    names = result.names
                    top1_idx = probs.top1
                    raw_label = names[top1_idx].upper()
                    confidence = float(probs.top1conf)

                    # Keputusan Pasti: Hanya RAPI atau BERANTAKAN (atau KOSONG)
                    if tot_s_live == 0 and raw_label == "KOSONG":
                        label = "KOSONG"
                    elif num_u_live > 0 or raw_label == "BERANTAKAN":
                        label = "BERANTAKAN"
                    else:
                        label = "RAPI"

                    # State machine
                    time_since_last_alert = curr_t - last_alert_time
                    in_cooldown = time_since_last_alert < cooldown_sec

                    if label == "BERANTAKAN":
                        if in_cooldown:
                            banner_color = (255, 140, 0)
                            rem = int(cooldown_sec - time_since_last_alert)
                            status_title = f"COOLDOWN: {rem}s"
                            status_sub = f"Tercecer: {num_u_live} | Peringatan dijeda"
                        else:
                            if messy_start_time is None:
                                messy_start_time = curr_t
                            dur = curr_t - messy_start_time
                            if dur >= trigger_sec:
                                banner_color = (0, 0, 220)
                                status_title = "PERINGATAN! SANDAL BERANTAKAN"
                                status_sub = f"{num_u_live} Sandal Tercecer | Alarm Berbunyi!"
                                last_alert_time = curr_t
                                messy_start_time = None
                                play_alert_sound()
                            else:
                                banner_color = (0, 165, 255)
                                status_title = f"TERDETEKSI BERANTAKAN ({dur:.1f}s / {trigger_sec:.1f}s)"
                                status_sub = f"Tercecer: {num_u_live} sandal | Memverifikasi..."
                    else:
                        messy_start_time = None
                        if label == "KOSONG":
                            banner_color = (220, 140, 20)
                            status_title = "STATUS: KOSONG / STANDBY"
                            status_sub = "Area bersih / tidak ada sandal"
                        else:
                            banner_color = (0, 200, 0)
                            status_title = "STATUS: RAPI"
                            status_sub = f"{num_p_live} Pasang Sandal Tertata Rapi"

                    # Draw overlay banner on annotated_frame (yang sudah memiliki kotak hijau/merah sandal)
                    h, w, _ = annotated_frame.shape
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 80), banner_color, -1)
                    cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)
                    cv2.putText(annotated_frame, status_title, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(annotated_frame, f"{status_sub} | FPS: {fps_live:.1f}", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    feed_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    with telemetry_status.container():
                        if label == "BERANTAKAN":
                            st.error(f"🚨 **{status_title}**")
                        elif label == "KOSONG":
                            st.info(f"🟦 **{status_title}**")
                        else:
                            st.success(f"✅ **{status_title}**")
                        st.caption(f"FPS: {fps_live:.1f} | Latensi: {1000/max(fps_live, 1):.1f} ms")

                    with telemetry_chart.container():
                        prob_dict_live = {names[i].upper(): float(probs.data[i]) for i in range(len(names))}
                        for c_k, c_v in sorted(prob_dict_live.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"**{c_k}**: {c_v*100:.1f}%")
                            st.progress(float(c_v))

            finally:
                cap.release()

# TAB 2: UJI SAMPEL DATASET
with tab_samples:
    st.subheader("Uji Sampel dari Dataset Validasi")
    st.write("Pilih salah satu gambar yang ada di folder validasi untuk langsung melihat hasil deteksi:")
    
    category = st.radio("Pilih Kategori Sampel:", ["BERANTAKAN", "RAPI", "KOSONG"], horizontal=True)
    cat_dir = os.path.join(VAL_DIR, category)
    
    if os.path.exists(cat_dir):
        file_list = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if file_list:
            selected_file = st.selectbox("Pilih File Sampel:", file_list)
            if selected_file:
                sample_path = os.path.join(cat_dir, selected_file)
                sample_img = Image.open(sample_path).convert("RGB")
                sample_pred = predict_pil_image(sample_img)
                st.markdown("---")
                render_prediction_card(sample_img, selected_file, sample_pred)
        else:
            st.warning("Tidak ditemukan file gambar pada folder kategori tersebut.")
    else:
        st.error(f"Folder validasi tidak ditemukan di `{cat_dir}`")

# TAB 3: METRIK MODEL & EVALUASI
with tab_metrics:
    st.subheader("Evaluasi Performa Model YOLOv8n-cls")
    st.markdown("""
    Model dilatih menggunakan dataset lokal selama 25 epoch dengan konfigurasi klasifikasi biner.
    Berikut hasil evaluasi performa pada data validasi:
    """)
    
    col_cm, col_curve = st.columns(2)
    
    cm_path = os.path.join(RUNS_DIR, "confusion_matrix_normalized.png")
    results_path = os.path.join(RUNS_DIR, "results.png")
    
    with col_cm:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix (Normalized)", width="stretch")
        else:
            st.info("Confusion matrix image tidak ditemukan.")
            
    with col_curve:
        if os.path.exists(results_path):
            st.image(results_path, caption="Kurva Training & Validation (Loss & Accuracy)", width="stretch")
        else:
            st.info("Grafik hasil training tidak ditemukan.")
            
    st.markdown("---")
    st.markdown("""
    ### 📈 Rangkuman Metrik:
    - **Top-1 Accuracy**: Mencapai **92.0%** pada epoch ke-4 hingga ke-17.
    - **Final Validation Loss**: ~0.26.
    - **Total Data Pelatihan**: 100 gambar (56 Berantakan, 44 Rapi).
    - **Total Data Validasi**: 25 gambar (14 Berantakan, 11 Rapi).
    """)
