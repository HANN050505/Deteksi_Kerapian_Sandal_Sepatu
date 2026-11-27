# Smart Shoe Monitor - Deteksi Sandal & Sepatu Laboratorium Teknik Informatika

Sistem Computer Vision berbasis **YOLOv8 Nano Classifier (3 Kelas: BERANTAKAN, RAPI, KOSONG)** yang diimplementasikan pada **Laboratorium Eksak / Laboratorium Komputer & Teknik Informatika (Lab TI)** untuk memantau kedisiplinan dan kerapian alas kaki praktikan/mahasiswa di depan pintu masuk lab secara *real-time* maupun melalui **Upload Foto**, serta memberikan peringatan audio otomatis berbahasa Indonesia jika sandal berserakan:
> *"Tolong rapikan sandal atau sepatu anda."*

---

## 📁 Struktur Direktori Proyek

```
deteksi_sandal/
│
├── core/                       <- Modul inti terpadu (config, motion watcher, inference)
│   ├── config.py               <- Parameter konfigurasi terpusat (single source of truth)
│   ├── motion.py               <- Event-driven motion settlement watcher & state machine
│   └── inference.py            <- Pipeline prediksi tunggal, spatial ensemble & letterbox
│
├── tools/                      <- Perangkat bantu dataset & audit integritas
│   ├── audit_dataset.py        <- Detektor duplikat (MD5) & near-duplicate (pHash)
│   └── split_dataset.py        <- Skrip Group Split 3-bagian bebas kebocoran data
│
├── reports/                    <- Laporan evaluasi empiris terukur (JSON & Confusion Matrix)
│   ├── evaluation_report.json  <- Metrik terukur, Wilson 95% CI & False Alarm Rate
│   └── confusion_matrix_eval.png
│
├── audio/
│   └── peringatan.mp3          <- Audio peringatan bahasa Indonesia
│
├── dataset/                    <- Dataset 3-split bebas kebocoran lintas grup
│   ├── train/                  <- 288 gambar (114 Berantakan, 100 Rapi, 74 Kosong)
│   ├── val/                    <- 69 gambar (27 Berantakan, 23 Rapi, 19 Kosong)
│   └── test/                   <- 68 gambar (26 Berantakan, 23 Rapi, 19 Kosong)
│
├── models/
│   ├── best.pt                 <- Bobot model terbaik 3-kelas (BERANTAKAN, RAPI, KOSONG)
│   └── best_v2_backup.pt       <- Cadangan bobot model versi terdahulu
│
├── app_upload.py               <- Dashboard Web interaktif (Streamlit)
├── evaluate.py                 <- Evaluator empiris (Wilson CI, FAR, komparasi ensemble)
├── realtime_detect.py          <- Aplikasi deteksi real-time webcam/video + threaded async
├── diagnostik_prototype.py     <- Script self-diagnostic objektif (PASS / WARN / FAIL)
├── train.py                    <- Script training YOLOv8n-cls 3 kelas
├── prepare_3class_dataset.py   <- Integrasi dataset Kosong (reproducible & idempotent)
│
├── jalankan_web_upload.bat     <- Sekali klik: Buka Dashboard Web
├── jalankan_deteksi.bat        <- Sekali klik: Jalankan deteksi webcam real-time
├── uji_diagnostik_prototype.bat <- Sekali klik: Uji kesehatan & kesiapan prototype
└── uji_foto.bat                <- Sekali klik: Uji satu foto via file dialog
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

## 🧠 Fitur Pintar & Logika Real-Time

1. **Dual Interface**: Mendukung pemantauan video stream kamera *real-time* sekaligus pengujian interaktif berbasis *upload foto*.
2. **Event-Driven Motion Settlement Watcher**:
   Sistem tidak mengevaluasi saat orang sedang melintas atau melepas sandal. Evaluasi cerdas dipicu tepat setelah pergerakan selesai (*settled* selama 1.8 detik), mengeliminasi *false alarm* secara drastis. Gerakan selama jeda cooldown tetap dicatat agar orang yang datang bergantian tidak terlewatkan.
3. **Threaded Asynchronous Inference**:
   Inferensi deep learning dijalankan pada antrean thread latar belakang (background worker) terpisah sehingga rendering kamera video tetap mulus pada 30+ FPS tanpa mengalami freeze/stutter.
4. **Temporal Voting & Confidence Threshold**:
   Peringatan suara hanya dipicu jika keyakinan model $\ge 70\%$ dan minimal 2 dari 3 evaluasi berurutan bernilai `BERANTAKAN`. Jika keyakinan di bawah ambang batas, status dialihkan ke `RAGU` dan audio dijeda demi mencegah alarm palsu.
5. **Anti-Spam (Smart Cooldown)**:
   Setelah peringatan suara berbunyi, sistem memberi waktu jeda cooldown (10 detik) agar praktikan dapat merapikan alas kakinya dengan tenang tanpa suara berulang-ulang.

---

## 🎯 Definisi Operasional Kelas

| Kelas | Kriteria Visual Utama | Kasus Batas (Edge Cases) |
|---|---|---|
| **`RAPI`** | Sandal/sepatu diletakkan berdampingan secara berpasangan (kiri dan kanan sejajar menghadap arah yang sama), tersusun teratur di rak sandal atau di pinggir dinding koridor. | Jika satu pasang sandal sedikit serong namun tidak menghalangi jalur keluar-masuk pintu dan tidak bertumpuk, kondisi tetap dinilai sebagai **RAPI**. |
| **`BERANTAKAN`** | Sandal terpisah jauh dari pasangannya, saling bertumpuk (*overlapping*), melintang acak di tengah pintu masuk lab, atau terbalik. | Jika sandal tertata berpasangan rapi namun diletakkan persis melintang di tengah pintu masuk (menghalangi K3 jalur evakuasi darurat), kondisi diklasifikasikan sebagai **BERANTAKAN**. |
| **`KOSONG`** | Area lantai pemantauan bersih tanpa alas kaki (kondisi standby saat praktikum berlangsung atau di luar jam praktikum). | Mode standby murni; audio peringatan tidak akan pernah berbunyi pada kondisi ini. |

---

## ⚠️ Batasan Sistem & Karakteristik Model

1. **Klasifikasi Citra, Bukan Deteksi Objek**:
   Model ini berbasis arsitektur **YOLOv8n-cls (Image Classifier)** yang mengklasifikasikan status keseluruhan area lantai menjadi satu label keputusan (`BERANTAKAN`, `RAPI`, atau `KOSONG`), **bukan** object detection yang menghasilkan koordinat bounding box per masing-masing sandal.
2. **Domain Shift (Kamera HP vs Webcam Real-Time)**:
   Sebagian besar dataset dilatih dari sudut pemotretan smartphone (~45° hingga ~75°). Pada implementasi webcam permanen di atas pintu, pastikan sudut kemiringan kamera diarahkan pada area fokus lantai (*Drop Zone*) agar distribusi fitur visual tetap konsisten.
3. **Kompatibilitas Sistem Operasi**:
   Sistem dioptimalkan untuk Windows menggunakan driver `cv2.CAP_DSHOW` untuk inisialisasi instan tanpa lag buffer. Skrip Python tetap kompatibel dengan Linux / macOS dengan menggunakan backend video default `cv2.CAP_ANY`.

---

## 🔒 Catatan Keamanan Jaringan & Privasi

1. **Akses Jaringan Web Streamlit**:
   Skrip peluncur `jalankan_web_upload.bat` secara default berjalan pada host lokal `127.0.0.1` demi keamanan. Menjalankan Streamlit dengan `--server.address 0.0.0.0` membuka akses antarmuka ke seluruh jaringan lokal tanpa autentikasi bawaan. Gunakan opsi tersebut secara sadar hanya di jaringan tepercaya.
2. **Privasi Penggunaan Kamera**:
   Kamera pemantau diarahkan khusus pada area lantai/rak alas kaki di depan pintu masuk. Pastikan pemasangan kamera di lingkungan publik telah memperoleh izin pengelola fasilitas serta tidak menyorot wajah atau aktivitas pribadi pengguna.

---

## 📖 Business & Data Understanding
Penjelasan lengkap mengenai latar belakang masalah, tujuan bisnis, metrik KPI, distribusi dataset, serta analisis kualitas data dapat dibaca pada file [BUSINESS_AND_DATA_UNDERSTANDING.md](BUSINESS_AND_DATA_UNDERSTANDING.md).
