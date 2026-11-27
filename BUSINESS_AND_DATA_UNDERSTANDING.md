# Dokumen Business Understanding & Data Understanding
## Proyek: Smart Shoe Monitor (Deteksi Sandal Rapi vs Berantakan)

Dokumen ini disusun mengikuti kerangka standar industri **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) untuk memberikan panduan komprehensif terkait latar belakang bisnis, tujuan, metrik, serta karakteristik data yang digunakan dalam model kecerdasan buatan deteksi kerapian sandal.

---

## 1. 🏢 BUSINESS UNDERSTANDING (Pemahaman Bisnis)

### 1.1 Latar Belakang & Pernyataan Masalah (Problem Statement)
Di lingkungan akademik perguruan tinggi, khususnya pada **Laboratorium Eksak / Laboratorium Komputer & Teknik Informatika (Lab TI)**, terdapat Standar Operasional Prosedur (SOP) yang mewajibkan seluruh mahasiswa, praktikan, dan asisten lab untuk melepas alas kaki luar (sandal atau sepatu) sebelum memasuki ruangan laboratorium. 

Kewajiban ini sangat krusial untuk:
- Menjaga kebersihan dan sterilitas ruangan ber-AC dari debu dan partikel kotoran luar yang dapat menyumbat sistem sirkulasi pendingin PC praktikum, switch jaringan, dan server laboratorium.
- Mencegah bahaya kelembaban dan kotoran basah yang berpotensi memicu korsleting listrik atau gangguan elektrostatik pada perangkat keras komputer.

Namun, kendala klasik yang rutin terjadi di depan pintu masuk laboratorium adalah:
1. **Semrawut Saat Pergantian Sesi Praktikum**: Pada jam-jam pergantian sif praktikum (ketika 30–50 mahasiswa keluar dan masuk secara bersamaan), sandal dan sepatu praktikan dilepas terburu-buru hingga berserakan dan bertumpuk tidak beraturan.
2. **Kesehatan & Keselamatan Kerja (K3) Laboratorium**: Tumpukan alas kaki yang acak menghalangi alur keluar-masuk pintu lab dan jalur evakuasi darurat, meningkatkan risiko praktikan tersandung.
3. **Risiko Tertukar & Hilang**: Alas kaki yang tercecer di lantai koridor rentan terinjak, rusak, atau tertukar antar mahasiswa.
4. **Beban Tambahan bagi Laboran & Asisten Praktikum**: Laboran (teknisi lab) maupun asisten praktikum harus berulang kali menegur mahasiswa atau merapikan alas kaki secara manual di sela-sela tugas teknis menyiapkan modul praktikum.

### 1.2 Tujuan Solusi (Business Objectives)
Sebagai representasi inovasi teknologi di bidang Informatika (*Smart Campus / Smart Lab*), sistem **Smart Shoe Monitor** dirancang untuk:
- Mengotomatisasi pemantauan kerapian area alas kaki di depan pintu masuk Laboratorium Teknik Informatika secara berkelanjutan (*real-time 24/7*) menggunakan webcam/CCTV.
- Memberikan intervensi perilaku (*behavioral nudge*) berupa teguran suara otomatis yang sopan (*"Tolong rapikan sandal atau sepatu anda"*) seketika saat kondisi sandal berantakan terdeteksi di area pemantauan.
- Mendorong kedisiplinan dan kesadaran mandiri (*self-correction*) mahasiswa/praktikan agar tertib meletakkan sandal pada rak atau area yang ditentukan tanpa perlu teguran tatap muka.
- Membebaskan laboran dan asisten praktikum dari rutinitas menertibkan sandal, sehingga mereka dapat fokus sepenuhnya pada pendampingan akademik praktikum dan pemeliharaan fasilitas IT.

### 1.3 Kriteria Keberhasilan & Matriks Penyelarasan (Terukur vs Target vs Asumsi)

Untuk menjaga kejujuran ilmiah dan ketepatan evaluasi teknik, metrik keberhasilan dipisahkan secara tegas ke dalam 3 kategori status:
- **Hasil Terukur (Empirical Measurement)**: Metrik yang diuji langsung pada *independent test set* ($N=68$) atau benchmarking riil perangkat.
- **Target (KPI Goal)**: Standar acuan keberhasilan yang ditetapkan dalam proposal proyek.
- **Asumsi / Simulasi**: Proyeksi hipotetis yang belum diverifikasi dengan studi longitudinal lapangan.

| Kategori | Parameter KPI | Target Acuan | Status & Hasil Nyata | Keterangan Status |
| :--- | :--- | :--- | :--- | :--- |
| **Teknis (Technical KPI)** | Top-1 Classification Accuracy | $\ge 90.0\%$ | **95.59%** (Ensemble) / **97.06%** (Single) | **Hasil Terukur** (Test Set Independen, $N=68$) |
| | Wilson 95% Confidence Interval | Rentang $\pm 10\%$ | **[87.8%, 98.5%]** (Ensemble) | **Hasil Terukur** (Wilson Score Interval) |
| | False Alarm Rate (RAPI $\rightarrow$ BER) | $< 10.0\%$ | **4.35%** (1/23 RAPI salah diprediksi) | **Hasil Terukur** (Ensemble + Threshold 70%) |
| | Latensi Pipeline Penuh (CPU) | $< 1500$ ms | **~720 - 1440 ms** (Ensemble 3-View) | **Hasil Terukur** (Worker Async terpisah, FPS kamera 30+) |
| **Dampak Lab (Impact KPI)** | Kepatuhan Praktikan (Compliance) | $\ge 75.0\%$ | *Belum Diukur di Lapangan* | **Asumsi / Target Lapangan** |
| | Efisiensi Kerja Staf/Laboran | Pengurangan intervensi | *Simulasi Hipotetis* (~15-30 mnt/hari) | **Asumsi / Simulasi Modeling** |

### 1.4 Batasan & Manajemen Risiko (Constraints & Risks)
1. **Lantai Bersih / Tanpa Sandal (Idle State)**:
   - *Tantangan*: Saat sesi praktikum berlangsung, area depan pintu lab seringkali kosong (tidak ada sandal atau sepatu).
   - *Mitigasi*: Menambahkan kelas **`KOSONG`** (didukung 22 citra unik lapangan) sehingga sistem berada dalam mode *standby* tanpa memicu audio peringatan.
2. **Perubahan Pencahayaan Koridor**:
   - *Mitigasi*: Augmentasi warna HSV dan RandAugment pada pelatihan model, serta penerapan ambang keyakinan (*confidence threshold* 70%).
3. **Objek Dinamis (Orang Melintas / Sedang Menaruh Sandal)**:
   - *Mitigasi*: Fitur *Event-Driven Motion Settlement Watcher* (sistem menunggu kondisi tenang selama **1.8 detik** setelah pergerakan selesai sebelum mengevaluasi citra). Gerakan selama cooldown tetap dicatat.
4. **Pencegahan Alarm Berulang (Anti-Spam)**:
   - *Mitigasi*: Fitur *Smart Cooldown* (jeda **10.0 detik**) agar praktikan memiliki waktu cukup untuk merapikan sandal tanpa terganggu audio berulang.

---

## 2. 📊 DATA UNDERSTANDING (Pemahaman Data)

### 2.1 Sumber & Metode Pengumpulan Data (Data Collection)
Dataset dihimpun melalui pengamatan dan dokumentasi visual langsung di lingkungan **Laboratorium Eksak / Teknik Informatika**:
- **Lokasi Observasi**: Area pintu masuk laboratorium komputer, koridor depan lab, rak sandal/sepatu praktikum, dan lantai ubin pembatas area laboratorium.
- **Kondisi Tanpa Sandal (`KOSONG`)**: Citra lantai ubin koridor lab dalam kondisi bersih tanpa alas kaki (total 22 citra asli unik).
- **Perangkat**: Kamera smartphone dengan orientasi bervariasi (sudut miring ~45° dan sudut atas ~75°).
- **Format Data**: Citra digital RGB (.jpg, .jpeg, .png).

### 2.2 Struktur & Distribusi Dataset Bebas Kebocoran (Group Split)
Data dibagi menggunakan skema **Group Split (terstratifikasi per sesi/mahasiswa/kelompok)** ke dalam **3 bagian** independen untuk mencegah kebocoran data (*data leakage*):

```
D:\deteksi_sandal\dataset\
├── train/ (288 citra, 67.8%)
│   ├── BERANTAKAN/  -> 114 citra
│   ├── KOSONG/      -> 74 citra (14 citra unik + augmentasi terukur)
│   └── RAPI/        -> 100 citra
├── val/ (69 citra, 16.2%)
│   ├── BERANTAKAN/  -> 27 citra
│   ├── KOSONG/      -> 19 citra (4 citra unik + variasi)
│   └── RAPI/        -> 23 citra
└── test/ (68 citra, 16.0% - SET EVALUASI INDEPENDEN)
    ├── BERANTAKAN/  -> 26 citra
    ├── KOSONG/      -> 19 citra (4 citra unik + variasi)
    └── RAPI/        -> 23 citra
Total: 425 citra (22 citra KOSONG unik lapangan)
```

- **Integritas Uji Bebas Kebocoran**: Berdasarkan audit hash perceptual (pHash $\le 4$) dan MD5, **0% kebocoran lintas split** tercapai antar train, val, dan test.
- **Kemandirian Test Set**: Test set bersifat independen murni dan tidak pernah disentuh dalam proses early stopping maupun pemilihan checkpoint model.

### 2.3 Karakteristik & Ciri Visual Tiap Kelas

#### A. Kelas BERANTAKAN:
- Sandal tidak berpasangan sejajar (kiri dan kanan terpisah jauh atau orientasi sudut berlawanan).
- Posisi bertumpuk satu sama lain (*overlapping*).
- Pola peletakan acak, miring, melintang di tengah jalur jalan, atau terbalik.
- Jarak antar-sandal tidak konsisten.

#### B. Kelas RAPI:
- Sandal diletakkan berdampingan secara berpasangan (kiri dan kanan sejajar menghadap arah yang sama).
- Posisi tertata di pinggir dinding, di rak sandal, atau mengikuti garis nat keramik.
- Tidak ada sandal yang bertumpuk atau melintang menghalangi jalan.

### 2.4 Evaluasi Kualitas Data & Tantangan (Data Quality Assessment)
1. **Ukuran Dataset Ringkas (*Dataset Overview*)**:
   - Total 425 gambar (357 data latih dan 68 data validasi) mencakup tiga kondisi operasional lapangan: BERANTAKAN, RAPI, dan KOSONG.
   - *Analisis Solusi*: Pendekatan arsitektur **YOLOv8n-cls** yang memanfaatkan bobot awal (*pre-trained on ImageNet*) terbukti efektif karena model telah memiliki fitur representasi tepi, tekstur, dan bentuk dasar sejak awal, sehingga mampu mencapai akurasi 92% tanpa mengalami overfitting parah.
2. **Variasi Background**:
   - Sebagian besar berlatar belakang lantai keramik putih atau bermotif abu-abu/krem. Sandal dengan warna kontras (merah, biru, hitam) sangat mudah dikenali.
3. **Resolusi Input**:
   - Model di-resize pada resolusi $224 \times 224$ piksel saat inferensi, yang sangat ideal untuk menjaga kecepatan komputasi tinggi pada perangkat CPU tanpa GPU diskrit.

### 2.5 Rekomendasi Peningkatan Dataset di Masa Depan
1. **Penambahan Data Kondisi Khusus**: Menambahkan sampel pencahayaan redup/malam hari dan kondisi basah setelah hujan.
2. **Evolusi ke Object Detection / Instance Segmentation**: Jika ingin menghitung jumlah pasang sandal yang berantakan secara eksak, dataset dapat dianotasi dengan bounding box (format YOLO Detection).
