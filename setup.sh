#!/bin/bash

echo "--- [1/5] Updating System & Installing Media Tools ---"
apt-get clean
apt update && apt install -y ffmpeg curl python3-pip libass-dev fonts-dejavu

echo "--- [2/5] Installing AI & Web Frameworks (Optimized) ---"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper --no-deps
pip install fastapi uvicorn pydantic opencv-python yt-dlp numpy tiktoken fonttools numba
pip install google-api-python-client google-auth-oauthlib google-auth-httplib2

echo "--- [3/5] Downloading Face Detection Models ---"
mkdir -p /root/vastclip/shorts
cd /root/vastclip
curl -L -o deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -L -o res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

echo "--- [4/5] Final Cleanup to Save Disk ---"
pip cache purge
rm -rf /root/.cache/pip

echo "--- [5/5] Setup Selesai! ---"
echo "Pastikan file main.py sudah di /root/vastclip/"
echo "Jalankan dengan: python3 main.py"