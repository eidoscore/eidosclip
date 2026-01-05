# EidosCLIP - AI-Powered Auto Clipper!

eidosclip adalah tools yang digunakan untuk membuat short video otomatis dari youtube video, Workflow **n8n** ini mengotomatiskan seluruh proses pembuatan konten Shorts dari podcast. Mulai dari submit transkrip, analisis AI, hingga rendering di GPU Cloud (**Vast.ai**) dan upload otomatis ke **YouTube**.

[Register Vast.ai Account](https://cloud.vast.ai/?ref_id=378254)

# Files

StackEdit stores your files in your browser, which means all your files are automatically saved locally and are accessible **offline!**

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

##

### 🛠️ Detail Alur Kerja Teknis

1.  **Input Collection**: User submit data via **n8n Form Submission**.
2.  **Server Handshake**: n8n mengambil IP instance **Vast.ai** yang sedang aktif secara dinamis.
3.  **NLP Analysis**: **DeepSeek API** membedah transkrip untuk mendapatkan _timestamp_ segmen terbaik.
4.  **DNN Processing**: Perintah render dikirim ke server; model **DNN** bekerja mendeteksi subjek untuk melakukan _smart crop_ 9:16.
5.  **Final Execution**: Server menghasilkan file video Shorts yang sudah terpotong sempurna dan siap didistribusikan.

##

## Apa yang Perlu Dikonfigurasi?

- **n8n Setup**: Import workflow `.json` dan sesuaikan node **Config Server** dengan API Key kamu.
- **DeepSeek API**: Masukkan API Key aktif kamu untuk proses segmentasi transkrip.
- **Vast.ai Instance**: Siapkan instance GPU dan pastikan skrip pemrosesan video (DNN/OpenCV) sudah terinstal untuk menerima perintah dari n8n.

## ⚙️ Prasyarat (Prerequisites)

- **n8n**: Instance aktif (Docker/Cloud).
- **Vast.ai Account**: Instance GPU yang sudah terinstal skrip renderer (ffmpeg/python).
- **DeepSeek API Key**: Untuk otak analisis segmen.
- **YouTube Data API v3**: Diaktifkan di Google Cloud Console.
  The file explorer is accessible using the button in left corner of the navigation bar. You can create a new file by clicking the **New file** button in the file explorer. You can also create folders by clicking the **New folder** button.

## 🚧 Roadmap (Fitur Mendatang)

- **Auto SEO & YouTube Upload**: Otomatisasi pembuatan judul, deskripsi, tags, dan upload ke channel.
- **Google Sheets Integration**: Pelacakan status pengerjaan dan database konten secara otomatis.
