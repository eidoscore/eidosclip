#!/bin/bash

# Mendeteksi lokasi script saat ini agar dinamis
BASE_DIR=$(pwd)

echo "--- [1/5] Updating System & Installing Media Tools ---"
apt-get clean
apt update && apt install -y ffmpeg curl python3-pip libass-dev fonts-dejavu

echo "--- [2/5] Installing PyTorch (CUDA Optimized) ---"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

echo "--- [3/5] Installing Dependencies from requirements.txt ---"
pip install -r requirements.txt

echo "--- [4/5] Preparing Project Structure & Models ---"
mkdir -p "$BASE_DIR/shorts"
mkdir -p "$BASE_DIR/models"

cd "$BASE_DIR/models"
echo "[>] Downloading Face Detection Models to $BASE_DIR/models..."
curl -L -o deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -L -o res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

echo "--- [5/5] Final Cleanup to Save Disk ---"
pip cache purge
rm -rf /root/.cache/pip

cd "$BASE_DIR"
echo "--- SETUP SELESAI! ---"
echo "Struktur Folder Sekarang:"
echo " - $BASE_DIR/main.py"
echo " - $BASE_DIR/models/ (Sudah berisi model dnn)"
echo " - $BASE_DIR/fonts/ (Silahkan copy file .ttf lo ke sini)"
echo " - $BASE_DIR/shorts/ (Folder output video)"
echo ""
echo "Jalankan dengan: python3 main.py"