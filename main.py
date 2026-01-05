import os, subprocess, glob, torch, whisper, cv2, json, re, time, sys, pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
from fastapi.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Google API Libraries
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

app = FastAPI()

# --- INISIALISASI ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
BASE_DIR = "/root/vastclip"
OUTPUT_DIR = os.path.join(BASE_DIR, "shorts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# KONFIGURASI FONT
FONT_NAME = "DejaVu Sans"
possible_fonts = ["Poppins-Bold.ttf", "Montserrat-Bold.ttf"]
for f_name in possible_fonts:
    if os.path.exists(os.path.join(BASE_DIR, f_name)):
        FONT_NAME = f_name.split("-")[0]
        break

# Load DNN Face Detector
PROTOTXT = "deploy.prototxt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_FILE)

BASE_DOWNLOAD_URL = ""
global_all_words = []

# --- MODELS ---
class MultiRenderRequest(BaseModel):
    segments: Any
    original_video_url: Optional[str] = None

class DirectUploadRequest(BaseModel):
    filename: str
    title: str
    channel_alias: str
    description: Optional[str] = ""
    tags: Optional[str] = ""

# --- YOUTUBE AUTH ENGINE ---
def get_youtube_service(channel_alias: str):
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None
    token_file = os.path.join(BASE_DIR, f'token_{channel_alias}.pickle')

    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    if creds and creds.expired and creds.refresh_token:
        try:
            print(f"[>] Refreshing token for alias: {channel_alias}")
            creds.refresh(Request())
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        except Exception as e:
            print(f"[!] Gagal refresh token: {e}")
            creds = None

    if not creds or not creds.valid:
        raise Exception(f"Kredensial {channel_alias} tidak ditemukan. Jalankan auth.py dulu.")

    return build("youtube", "v3", credentials=creds)

# --- HELPER FUNCTIONS ---
def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text

def to_ass_time(sec):
    sec = max(0, sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{int(h):01d}:{int(m):02d}:{int(s):02d}.{int((s-int(s))*100):02d}"

def cleanup_raw_files():
    """Menghapus file mentah setelah proses selesai"""
    files_to_delete = ["raw_video.mp4", "audio.wav"]
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
            print(f"[!] Cleaned up raw file: {f}")

def create_segment_subtitle(words, start_offset, end_offset, out_path):
    header = (
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginV\n"
        f"Style: Default,{FONT_NAME},105,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,1,6,1,2,420\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Text\n"
    )
    seg_words = [w for w in words if w['start'] >= (start_offset - 0.2) and w['end'] <= (end_offset + 0.2)]
    lines = []
    for i in range(0, len(seg_words), 3):
        chunk = seg_words[i:i+3]
        s_t = to_ass_time(chunk[0]['start'] - start_offset)
        e_t = to_ass_time(chunk[-1]['end'] - start_offset)
        karaoke = ""
        for w in chunk:
            dur = max(1, int((w['end'] - w['start']) * 100))
            clean = re.sub(r"[^A-Z0-9' ]", "", w['word'].strip().upper())
            karaoke += f"{{\\k{dur}}}{clean} "
        lines.append(f"Dialogue: 0,{s_t},{e_t},Default,{karaoke.strip()}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines))

def get_smart_crop_params(video_path, start_time, end_time, frame_width, frame_height):
    try:
        cap = cv2.VideoCapture(video_path)
        detected_centers_x = []
        check_points = np.linspace(start_time, end_time, num=10)
        for t in check_points:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret: continue
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                if detections[0, 0, i, 2] > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, _, endX, _) = box.astype("int")
                    detected_centers_x.append(startX + (endX - startX) // 2)
                    break
        cap.release()
        target_w = min(int(frame_height * (9/16) * 1.2), frame_width)
        if detected_centers_x:
            avg_x = sum(detected_centers_x) / len(detected_centers_x)
            start_x = max(0, min(frame_width - target_w, avg_x - (target_w // 2)))
            return int(start_x), target_w
        return (frame_width - target_w) // 2, target_w
    except: return (frame_width - 960) // 2, 960

# --- ENDPOINTS ---
@app.on_event("startup")
async def startup_event():
    global BASE_DOWNLOAD_URL
    try:
        ip_address = subprocess.check_output(["curl", "-s", "ifconfig.me"]).decode().strip()
        BASE_DOWNLOAD_URL = f"http://{ip_address}:8000/download_short"
        print(f"--- SERVER READY: {BASE_DOWNLOAD_URL} ---")
    except: pass

@app.post("/process")
def process_video(req: dict):
    global global_all_words
    try:
        url = req.get("url")
        # Hapus sisa file lama jika ada
        cleanup_raw_files()

        subprocess.run(["yt-dlp", "-f", "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best", "--merge-output-format", "mp4", url, "-o", "raw_video.mp4"], check=True)
        subprocess.run(["ffmpeg", "-i", "raw_video.mp4", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "audio.wav", "-y"], check=True, capture_output=True)

        result = model.transcribe("audio.wav", word_timestamps=True, fp16=True, verbose=False)
        global_all_words = []
        full_transcript = ""
        for i, segment in enumerate(result['segments']):
            full_transcript += f"[{segment['start']:.2f} --> {segment['end']:.2f}] {segment['text']}\n"
            if 'words' in segment: global_all_words.extend(segment['words'])

        return {"status": "success", "transcript": full_transcript}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/render_multi")
def render_multi(req: MultiRenderRequest):
    global global_all_words
    try:
        data = req.segments
        if isinstance(data, str): data = json.loads(data)

        video_info = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", "raw_video.mp4"]).decode().strip()
        orig_w, orig_h = map(int, video_info.split('x'))

        def render_worker(i, s_dict):
            try:
                idx = i + 1
                start, end = float(s_dict['start']), float(s_dict['end'])
                raw_title = s_dict.get('title', f"short_{idx}")
                clean_title = slugify(raw_title)
                sub_file = f"sub_{idx}.ass"
                create_segment_subtitle(global_all_words, start, end, sub_file)
                sub_path = os.path.abspath(sub_file).replace(":", "\\:")
                smart_x, crop_w = get_smart_crop_params("raw_video.mp4", start, end, orig_w, orig_h)
                out_filename = f"{clean_title}.mp4"
                out_path = os.path.join(OUTPUT_DIR, out_filename)

                cmd = [
                    "ffmpeg", "-y", "-ss", str(start), "-t", str(end-start),
                    "-i", "raw_video.mp4",
                    "-vf", f"setpts=PTS-STARTPTS,crop={crop_w}:ih:{smart_x}:0,scale=1080:1920,subtitles='{sub_path}':fontsdir='{BASE_DIR}/':force_style='Fontname={FONT_NAME},Alignment=2,MarginV=420'",
                    "-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "6M",
                    "-c:a", "aac", "-b:a", "192k", out_path
                ]
                subprocess.run(cmd, check=True)
                if os.path.exists(sub_file): os.remove(sub_file)
                return {"status": "SUCCESS", "filename": out_filename, "url": f"{BASE_DOWNLOAD_URL}/{out_filename}"}
            except Exception as e:
                return {"status": "error", "message": str(e), "index": i}

        results = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(render_worker, i, s) for i, s in enumerate(data)]
            for future in as_completed(futures): results.append(future.result())

        # --- PANGGIL CLEANUP SETELAH RENDER SELESAI ---
        cleanup_raw_files()

        return {"status": "all_completed", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/upload_direct")
async def upload_direct(req: DirectUploadRequest):
    try:
        video_path = os.path.join(OUTPUT_DIR, req.filename)
        if not os.path.exists(video_path):
            return {"status": "error", "message": "File video tidak ditemukan"}

        youtube = get_youtube_service(req.channel_alias)

        # 1. Rapikan Deskripsi
        desc = req.description if req.description else ""
        if "#shorts" not in desc.lower():
            desc = f"{desc}\n\n\n#shorts"
        else:
            desc = desc.replace("#shorts", "\n\n\n#shorts")

        # 2. Rapikan Tags (Hapus duplikat dan bersihkan spasi)
        raw_tags = [t.strip() for t in req.tags.split(",")] if req.tags else []
        final_tags = list(dict.fromkeys([t for t in raw_tags if t.lower() != "shorts"]))

        request_body = {
            'snippet': {
                'title': req.title[:100],
                'description': desc,
                'tags': final_tags,
                'categoryId': '22'
            },
            'status': {
                'privacyStatus': 'public',
                'selfDeclaredMadeForKids': False
            }
        }

        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        print(f"\n[>] UPLOADING {req.filename} TO YOUTUBE: {req.channel_alias}")
        response = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media).execute()

        # 3. Hapus file video lokal SETELAH upload sukses
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"[!] File deleted after successful upload: {req.filename}")

        return {"status": "success", "youtube_id": response.get("id"), "link": f"https://youtu.be/{response.get('id')}"}
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/download_short/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(path) if os.path.exists(path) else {"error": "not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
