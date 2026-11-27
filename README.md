# Smart Shoe Monitor - Deteksi Sandal & Sepatu Laboratorium Teknik Informatika

Sistem Computer Vision berbasis **YOLOv8 Nano Classifier (3 Kelas: BERANTAKAN, RAPI, KOSONG)** yang diimplementasikan pada **Laboratorium Eksak / Laboratorium Komputer & Teknik Informatika (Lab TI)** untuk memantau kedisiplinan dan kerapian alas kaki praktikan/mahasiswa di depan pintu masuk lab secara *real-time* maupun melalui **Upload Foto**, serta memberikan peringatan audio otomatis berbahasa Indonesia jika sandal berserakan:
> *"Tolong rapikan sandal atau sepatu anda."*

---

## 📁 Struktur Direktori Proyek

```
deteksi_sandal/
│
├── audio/
│   └── peringatan.mp3          <- Audio peringatan bahasa Indonesia
│
├── dataset/
│   ├── train/
│   │   ├── BERANTAKAN/         <- 133 gambar
│   │   ├── KOSONG/             <- 108 gambar (tanpa alas kaki)
│   │   └── RAPI/               <- 116 gambar
│   └── val/
│       ├── BERANTAKAN/         <- 34 gambar
│       ├── KOSONG/             <- 4 gambar
│       └── RAPI/               <- 30 gambar
│
├── models/
│   ├── best.pt                 <- Bobot model terbaik 3-kelas (BERANTAKAN, RAPI, KOSONG)
│   └── best_v2_backup.pt       <- Cadangan bobot model versi 2
│
├── runs/                       <- Log dan grafik metrik hasil training
│
├── app_upload.py               <- Dashboard Web interaktif (Streamlit) untuk upload foto & kamera HP
├── predict_image.py            <- Script desktop pengujian foto (File picker / CLI)
├── realtime_detect.py          <- Aplikasi deteksi real-time webcam/video + audio alert
├── diagnostik_prototype.py     <- Script self-diagnostic & verifikasi kesiapan prototype
├── train.py                    <- Script training YOLOv8n-cls 3 kelas
├── prepare_3class_dataset.py   <- Script integrasi dataset Kosong.zip & augmentasi 3-kelas
├── prepare_combined_dataset.py <- Script penggabungan dataset lama
├── generate_audio.py           <- Generator audio gTTS
│
├── jalankan_web_upload.bat     <- Sekali klik: Buka Dashboard Web Upload Foto
├── uji_foto.bat                <- Sekali klik: Pilih foto dari komputer & uji
├── jalankan_deteksi.bat        <- Sekali klik: Jalankan deteksi webcam real-time
├── uji_diagnostik_prototype.bat <- Sekali klik: Uji kesehatan & kesiapan lengkap prototype
│
├── Business_and_Data_Understanding.ipynb <- Notebook Jupyter interaktif (Code, EDA & Simulasi ROI)
└── BUSINESS_AND_DATA_UNDERSTANDING.md   <- Dokumen teks Business & Data Understanding
```

---

## 📦 Instalasi Dependensi

Pastikan Python 3.9+ telah terpasang, lalu pasang seluruh dependensi dengan:
```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan

### 1. Menguji Melalui Dashboard Web (Rekomendasi - Termasuk Kamera Real-Time & HP)
Klik dua kali file:
```
jalankan_web_upload.bat
```
Aplikasi web Streamlit akan terbuka di browser:
- **Di Laptop**: `http://localhost:8501`
- **Di HP (iOS / Android)**: Buka `Network URL` yang tertera pada jendela terminal (contoh: `http://192.168.x.x:8501`), pastikan HP dan laptop terhubung ke Wi-Fi / Hotspot yang sama.

**Fitur Dashboard:**
- 📤 **Upload & Ambil Foto Langsung dari HP**: Klik tombol *Browse files* di HP untuk langsung memotret lantai lab menggunakan kamera HP asli (tanpa izin browser yang ribet) atau pilih dari galeri.
- 📹 **Deteksi Kamera Real-Time**: Streaming video live langsung di browser dengan webcam laptop / kamera USB.
- 🧪 **Uji Sampel Validasi**: Mencoba sampel citra `BERANTAKAN`, `RAPI`, dan `KOSONG`.
- 📊 **Metrik Evaluasi Model**: Grafik kurva training loss, akurasi, dan normalized confusion matrix.

---

### 2. Menguji Melalui Dialog File Foto (Desktop)
Klik dua kali file:
```
uji_foto.bat
```
Jendela dialog file Windows akan terbuka untuk memilih gambar, lalu menampilkan hasilnya di layar beserta peringatan audio bila sandal berantakan.

Atau lewat command prompt:
```bash
python predict_image.py --image "path\ke\foto_sandal.jpg"
```

---

### 3. Menjalankan Deteksi Real-Time Webcam
Klik dua kali file:
```
jalankan_deteksi.bat
```
Atau lewat command prompt:
```bash
python realtime_detect.py --source 0
```

---

### 4. Menjalankan Diagnostik Mandiri Prototype
Klik dua kali file:
```
uji_diagnostik_prototype.bat
```
Atau lewat command prompt:
```bash
python diagnostik_prototype.py
```

---

## 🧠 Fitur Pintar

1. **Dual Interface**: Mendukung pemantauan video stream kamera *real-time* sekaligus pengujian interaktif berbasis *upload foto*.
2. **Event-Driven Motion Settlement Watcher**:
   Sistem tidak mengevaluasi saat orang sedang melintas atau melepas sandal. Evaluasi cerdas dipicu tepat setelah pergerakan selesai (*settled*), mengeliminasi *false alarm* secara drastis.
3. **Anti-Spam (Smart Cooldown)**:
   Setelah peringatan suara berbunyi, sistem memberi waktu jeda agar orang yang merapikan tidak terganggu suara berulang.
4. **Non-Blocking Multi-threaded Audio**:
   Aplikasi tetap responsif dan lancar tanpa mengalami freeze saat audio diputar.

---

## 🔒 Catatan Keamanan Jaringan & Privasi

1. **Akses Jaringan Web Streamlit**:
   Skrip peluncur `jalankan_web_upload.bat` secara default berjalan pada host lokal `127.0.0.1` demi keamanan. Menjalankan Streamlit dengan `--server.address 0.0.0.0` membuka akses antarmuka ke seluruh jaringan lokal tanpa autentikasi bawaan. Gunakan opsi tersebut secara sadar hanya di jaringan tepercaya.
2. **Privasi Penggunaan Kamera**:
   Kamera pemantau diarahkan khusus pada area lantai/rak alas kaki di depan pintu masuk. Pastikan pemasangan kamera di lingkungan publik telah memperoleh izin pengelola fasilitas serta tidak menyorot wajah atau aktivitas pribadi pengguna.

---

## 📖 Business & Data Understanding
Penjelasan lengkap mengenai latar belakang masalah, tujuan bisnis, metrik KPI, distribusi dataset, serta analisis kualitas data dapat dibaca pada file [BUSINESS_AND_DATA_UNDERSTANDING.md](BUSINESS_AND_DATA_UNDERSTANDING.md).
