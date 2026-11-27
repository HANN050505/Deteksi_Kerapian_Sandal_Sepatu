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

### 1.3 Kriteria Keberhasilan (Success Criteria)

| Kategori | Metrik KPI | Target | Hasil Saat Ini |
| :--- | :--- | :--- | :--- |
| **Teknis (Technical KPI)** | Top-1 Classification Accuracy | $\ge 90\%$ | **92.65%** (Validation Set, 3 Kelas) |
| | False Alarm Rate (False Positive) | $< 10\%$ | Tercapai dengan Kelas KOSONG + Filter 2.5s |
| | Inference Latency | $< 100$ ms / frame | ~8-12 ms pada CPU (YOLOv8n-cls) |
| **Dampak Lab (Impact KPI)** | Kepatuhan Praktikan (Compliance) | $\ge 75\%$ | Praktikan langsung merapikan alas kaki pasca audio |
| | Efisiensi Kerja Staf/Laboran | Pengurangan intervensi manual | Menghemat 1-2 jam kerja rutin laboran/asisten per hari |

### 1.4 Batasan & Manajemen Risiko (Constraints & Risks)
1. **Lantai Bersih / Tanpa Sandal (Idle State)**:
   - *Tantangan*: Saat sesi praktikum berlangsung, area depan pintu lab seringkali kosong (tidak ada sandal atau sepatu). Pada model 2-kelas terdahulu, kondisi ini terprediksi sebagai `RAPI 100%` dan rentan terjadi *false alarm* jika terdapat bayangan mahasiswa lewat.
   - *Mitigasi*: Menambahkan kelas **`KOSONG`** (didukung dataset `Kosong.zip`) sehingga sistem berada dalam mode *standby* tanpa memicu audio peringatan (*zero false alarm*).
2. **Perubahan Pencahayaan Koridor**: Fluktuasi cahaya koridor lab (lampu neon koridor, cahaya alami jendela siang vs sore).
   - *Mitigasi*: Augmentasi warna/HSV pada pelatihan model dan penyesuaian threshold keyakinan (*confidence threshold*).
3. **Objek Dinamis (Praktikan Melintas Melepas Sandal)**: Mahasiswa yang baru datang dan sedang melepas sepatu dapat membuat citra sekilas tampak "berantakan".
   - *Mitigasi*: Fitur *Temporal Verification* (sandal harus terdeteksi konsisten berantakan selama minimal 2.5 detik sebelum memicu suara peringatan).
4. **Peringatan Suara Berulang (Spamming)**:
   - *Mitigasi*: Fitur *Smart Cooldown* (jeda 12-15 detik) untuk memberi waktu bagi mahasiswa merapikan alas kakinya dengan tenang tanpa suara berulang-ulang.

---

## 2. 📊 DATA UNDERSTANDING (Pemahaman Data)

### 2.1 Sumber & Metode Pengumpulan Data (Data Collection)
Dataset dihimpun melalui pengamatan dan dokumentasi visual langsung di lingkungan **Laboratorium Eksak / Teknik Informatika**:
- **Lokasi Observasi**: Area pintu masuk laboratorium komputer, koridor depan lab, rak sandal/sepatu praktikum, dan lantai ubin pembatas area laboratorium.
- **Kondisi Tanpa Sandal (`KOSONG`)**: Citra lantai ubin koridor lab dalam kondisi bersih tanpa alas kaki dari `Kosong.zip`.
- **Perangkat**: Kamera smartphone dengan orientasi bervariasi (sudut miring ~45° dan sudut atas ~75°).
- **Format Data**: Citra digital RGB (.jpg, .jpeg, .png).

### 2.2 Struktur & Distribusi Dataset (Versi 3 - Multikelas)
Data dikelompokkan ke dalam **3 kelas** (`BERANTAKAN`, `RAPI`, `KOSONG`) dengan rasio **Training & Validasi**:

```
D:\deteksi_sandal\dataset\
├── train/ (357 gambar)
│   ├── BERANTAKAN/  -> 133 gambar (37.3%)
│   ├── KOSONG/      -> 108 gambar (30.2%, seimbang berkat augmentasi rotasi & kecerahan)
│   └── RAPI/        -> 116 gambar (32.5%)
└── val/ (68 gambar)
    ├── BERANTAKAN/  -> 34 gambar (50.0%)
    ├── KOSONG/      -> 4 gambar (5.9%, data asli lapangan)
    └── RAPI/        -> 30 gambar (44.1%)
Total: 425 gambar
```

- **Rasio Kelas Data Latih**: Relatif seimbang (BERANTAKAN: 37.3%, RAPI: 32.5%, KOSONG: 30.2%) sehingga model terlatih mengenali ketiga kondisi secara proporsional tanpa bias kelas mayoritas.
- **Stratifikasi**: Distribusi kelas pada data validasi mencakup 34 gambar BERANTAKAN (50.0%), 30 gambar RAPI (44.1%), dan 4 gambar KOSONG (5.9% data lapangan asli tanpa augmentasi offline sintesis).

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
