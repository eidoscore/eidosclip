# EidosCLIP - AI-Powered Auto Clipper!

![enter image description here](https://avatars.githubusercontent.com/u/40323224?v=4)
eidosclip adalah tools yang digunakan untuk membuat short video otomatis dari youtube video, Workflow **n8n** ini mengotomatiskan seluruh proses pembuatan konten Shorts dari podcast. Mulai dari submit transkrip, analisis AI, hingga rendering di GPU Cloud (**Vast.ai**) dan upload otomatis ke **YouTube**.

[Register Vast.ai Account](https://cloud.vast.ai/?ref_id=378254)

## Fitur Baru (v2.0)

- **✅ Google Sheets Database**: Sinkronisasi otomatis data segmen ke Spreadsheet. Pantau status render dan upload secara real-time.
- **✅ Auto YouTube Upload**: Upload otomatis ke YouTube Shorts lengkap dengan judul, deskripsi dinamis, dan 10+ metadata tags per video.
- **✅ Persistent OAuth System**: Menggunakan sistem `.pickle` token yang bisa melakukan _auto-refresh_, sehingga server bisa upload selamanya tanpa login ulang.

- **✅ Spreadsheet Centralized Control**: Kelola ratusan link video hanya dari satu lembar Google Sheets.

- **✅ Full Metadata SEO**: Tag YouTube terisi otomatis secara rapi per segmen (seperti: _Dune, fear, inspirasi, mindset_) sesuai hasil analisis AI.

- **✅ Smart Storage Management**: File `raw_video.mp4` dan `audio.wav` otomatis dihapus setelah proses selesai untuk menjaga storage tetap lega.

## 🚀 Fitur Utama

- **DNN-Powered Face Tracking (Smart Crop)**: Menggunakan teknologi **Deep Neural Networks (DNN)** di sisi server Vast.ai untuk mendeteksi koordinat wajah secara real-time. Sistem ini secara otomatis melakukan _dynamic cropping_ dari format landscape ke portrait dengan memastikan subjek utama selalu berada di tengah frame (center-focused).
- **AI Contextual Segmentation**: Mengintegrasikan **DeepSeek API** sebagai otak penganalisis transkrip untuk menentukan segmen mana yang paling memiliki nilai viralitas tinggi berdasarkan emosi dan bobot pembicaraan.
- **Automated Headless Pipeline**: Seluruh proses dari input form hingga pemrosesan DNN dilakukan tanpa intervensi manual (_hands-free_), diatur sepenuhnya oleh n8n sebagai _orchestrator_.
- **Native n8n Form Interface**: Antarmuka input yang efisien untuk memasukkan transkrip dan data podcast tanpa memerlukan _third-party_ form.
- **Dynamic Instance IP Discovery**: Workflow ini secara otomatis mendeteksi IP aktif dari instance **Vast.ai** kamu melalui node **Get Instance IP**, memastikan perintah render selalu terkirim ke alamat yang benar.
- **Centralized Environment Config**: Menggunakan node **Config Server** untuk manajemen variabel global seperti API keys, IP server, dan setting teknis lainnya dalam satu tempat.
- **Asynchronous Processing Pipeline**: Memisahkan tahap analisis teks (**DeepSeek API**) dengan tahap eksekusi video (**HTTP Request Render**) sehingga proses berjalan secara sistematis.
- **Stateless Workflow Design**: Setiap pengiriman data melalui form dianggap sebagai _item_ baru, memungkinkan skalabilitas jika ingin memproses banyak podcast sekaligus.
- **API-Driven GPU Orchestration**: Mengontrol server GPU di cloud secara jarak jauh hanya melalui HTTP request, memungkinkan rendering video resolusi tinggi dilakukan secara _headless_.
- **Smart Auto-Crop (AI-Driven)**: Menggunakan algoritma cerdas di sisi server (Vast.ai) untuk secara otomatis mendeteksi subjek utama (wajah pembicara) dan melakukan cropping video landscape menjadi portrait (9:16) tanpa memotong bagian penting.
- **Dynamic Clipping Engine**: Teknologi pemotongan video presisi tinggi berdasarkan _timestamp_ yang dihasilkan oleh DeepSeek AI, memastikan transisi antar segmen terasa halus dan natural.
- **Automated Subtitle Burning**: Proses otomatis menyematkan (burn-in) subtitle ke dalam video menggunakan data transkrip, meningkatkan retensi penonton pada platform Shorts/TikTok.
- **GPU-Accelerated Transcoding**: Memanfaatkan infrastruktur GPU dari **Vast.ai** untuk melakukan _encoding_ video secara paralel, sehingga proses render segmen berdurasi panjang bisa selesai dalam hitungan detik.
- **Intelligent Silence Removal**: Kemampuan untuk mendeteksi dan membuang bagian hening (silence) yang tidak perlu, sehingga video hasil akhir terasa lebih padat dan "fast-paced".
-

## 🔄 New Workflow: Spreadsheet-Driven

Sistem sekarang bekerja dengan flow yang lebih terorganisir:

1.  **Input**: Masukkan URL YouTube ke kolom yang ditentukan di **Google Sheets**.
2.  **Trigger**: n8n membaca baris baru dari spreadsheet secara berkala.
3.  **Process**: Script Python di Vast.ai mendownload, mentranskrip (Whisper), dan memotong video.
4.  **Database Update**: n8n mengupdate status pengerjaan dan link hasil render kembali ke spreadsheet.
5.  **Distribution**: Video yang sudah siap otomatis diupload ke channel YouTube yang dituju.

## 📁 Struktur Project (EidosCLIP)

Berdasarkan setup terbaru, berikut adalah susunan file utama di `/root/vastclip`:

- `setup.sh`: Script bash untuk instalasi otomatis seluruh tools.
- `main.py`: Core API server yang menghandle rendering dan upload.
- `token_xxxx.pickle`: Token akses YouTube yang persisten.
- `client_secrets.json`: Kredensial OAuth dari Google Cloud Console.
- `deploy.prototxt` & `res10_300...caffemodel`: Model DNN untuk deteksi wajah (Smart Crop).
- `shorts/`: Folder penyimpanan hasil akhir video sebelum diupload.

##

## Apa yang Perlu Dikonfigurasi?

- **n8n Setup**: Import workflow `.json` dan sesuaikan node **Config Server** dengan API Key kamu.
- **DeepSeek API**: Masukkan API Key aktif kamu untuk proses segmentasi transkrip.
- **Vast.ai Instance**: Siapkan instance GPU dan pastikan skrip pemrosesan video (DNN/OpenCV) sudah terinstal untuk menerima perintah dari n8n.

## ⚙️ Prasyarat (Prerequisites)

- **GPU Nvidia RTX (3090, 4090, atau A-series).**
- **n8n**: Instance aktif (Docker/Cloud).
- **Vast.ai Account**: Instance GPU yang sudah terinstal skrip renderer (ffmpeg/python).
- **DeepSeek API Key**: Untuk otak analisis segmen.
- **YouTube Data API v3**: Diaktifkan di Google Cloud Console.
  The file explorer is accessible using the button in left corner of the navigation bar. You can create a new file by clicking the **New file** button in the file explorer. You can also create folders by clicking the **New folder** button.

## Instalasi Cepat

Gunakan bash script yang sudah disediakan untuk mempersiapkan server dalam hitungan menit:

Bash

```
# 1. Clone & Masuk Folder
git clone https://github.com/eidoscore/eidosclip.git
cd eidosclip

# 2. Jalankan Auto-Installer
chmod +x setup.sh
./setup.sh

# 3. Jalankan API Server
python3 main.py
```

_Tips: Gunakan `screen` atau `pm2` agar script tetap jalan di background._

### 2. Konfigurasi Google Cloud (YouTube)

- Aktifkan **YouTube Data API v3**.
- Buat **OAuth 2.0 Client ID** (Web Application).
- Tambahkan `http://localhost:8080/` ke _Authorized redirect URIs_.
- Download `client_secret.json` dan taruh di folder root project.

### 3. Autentikasi YouTube (Pertama Kali)

Jalankan script pancingan token di terminal Vast.ai:

Bash

```
python3 auth.py

```

Ikuti instruksi yang muncul untuk menghasilkan file `token_eidosfinance.pickle`.

### 4. n8n Configuration

- Import file `workflow.json` ke n8n.
- Sesuaikan node **Config Server** dengan:

  - `DeepSeek API Key`
  - `Vast.ai IP & Port`
  - `Google Sheets ID`

## Persiapan Google Sheets & YouTube

1.  **n8n Setup**: Hubungkan node Google Sheets ke spreadsheet lu. Pastikan kolom `URL`, `Status`, dan `YouTube Link` sudah tersedia.
2.  **OAuth Auth**: Jalankan `python3 auth.py` sekali saja untuk memancing file `.pickle` agar fitur auto-upload jalan selamanya.
3.  **Metadata**: AI akan otomatis mengisi bagian **Tag** dengan kata kunci relevan untuk meningkatkan SEO video lu di Shorts Feed.

## 🚧 Roadmap (Fitur Mendatang)

- [ ] **TikTok & Instagram Auto-Post**: Ekspansi distribusi konten ke multi-platform.
- [ ] **Multiple Account Support**: Manajemen banyak channel YouTube dalam satu dashboard n8n.
- [ ] **AI Voiceover Overlay**: Menambahkan narasi suara AI jika audio asli kurang jelas.
