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

# --- INISIALISASI DINAMIS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
OUTPUT_DIR = os.path.join(BASE_DIR, "shorts")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PENDING_DIR = os.path.join(BASE_DIR, "pending_uploads")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PENDING_DIR, exist_ok=True)

# FORCE CPU MODE
device = "cpu"
print("[>] Running in CPU-ONLY mode.")
# Whisper menggunakan CPU, fp16 diset False karena CPU tidak mendukung fp16 secara native
model = whisper.load_model("base", device=device)

# --- KONFIGURASI FONT ---
FONT_NAME = "DejaVu Sans"
all_fonts = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
if all_fonts:
    montserrat_match = [f for f in all_fonts if "montserrat" in os.path.basename(f).lower()]
    target_font_path = montserrat_match[0] if montserrat_match else all_fonts[0]
    file_name = os.path.basename(target_font_path)
    FONT_NAME = file_name.replace(".ttf", "").split("-")[0].split("_")[0]

# --- LOAD DNN FACE DETECTOR ---
PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
MODEL_FILE = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_FILE) if os.path.exists(PROTOTXT) else None

BASE_DOWNLOAD_URL = ""
global_all_words = []

class MultiRenderRequest(BaseModel):
    segments: Any
    original_video_url: Optional[str] = None

class DirectUploadRequest(BaseModel):
    filename: str
    title: str
    channel_alias: str
    description: Optional[str] = ""
    tags: Optional[str] = ""

def get_youtube_service(channel_alias: str):
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None
    token_file = os.path.join(BASE_DIR, f'token_{channel_alias}.pickle')
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        except:
            creds = None
    return build("youtube", "v3", credentials=creds)

def core_upload_engine(video_path, data: DirectUploadRequest):
    youtube = get_youtube_service(data.channel_alias)
    desc = f"{data.description}\n\n#shorts" if "#shorts" not in data.description.lower() else data.description
    request_body = {
        'snippet': {'title': data.title[:100], 'description': desc, 'tags': data.tags.split(",") if data.tags else [], 'categoryId': '22'},
        'status': {'privacyStatus': 'public', 'selfDeclaredMadeForKids': False}
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    return youtube.videos().insert(part="snippet,status", body=request_body, media_body=media).execute()

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
    files_to_delete = [os.path.join(BASE_DIR, "raw_video.mp4"), os.path.join(BASE_DIR, "audio.wav")]
    for f in files_to_delete:
        if os.path.exists(f): os.remove(f)

def create_segment_subtitle(words, start_offset, end_offset, out_path):
    header = (
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginV\n"
        f"Style: Default,{FONT_NAME},95,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,1,6,1,2,450\n\n"
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
    with open(out_path, "w", encoding="utf-8") as f: f.write(header + "\n".join(lines))

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
        target_w = min(int(frame_height * (9/16)), frame_width)
        if detected_centers_x:
            avg_x = sum(detected_centers_x) / len(detected_centers_x)
            start_x = max(0, min(frame_width - target_w, avg_x - (target_w // 2)))
            return int(start_x), target_w
        return (frame_width - target_w) // 2, target_w
    except: return (frame_width - 960) // 2, 960

@app.on_event("startup")
async def startup_event():
    global BASE_DOWNLOAD_URL
    try:
        ip_address = subprocess.check_output(["curl", "-s", "ifconfig.me"]).decode().strip()
        BASE_DOWNLOAD_URL = f"http://{ip_address}:8000/download_short"
        print(f"--- CPU SERVER READY: {BASE_DOWNLOAD_URL} ---")
    except: pass

@app.post("/process")
def process_video(req: dict):
    global global_all_words
    try:
        url = req.get("url")
        cleanup_raw_files()
        video_raw = os.path.join(BASE_DIR, "raw_video.mp4")
        audio_raw = os.path.join(BASE_DIR, "audio.wav")
        subprocess.run(["yt-dlp", "-f", "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best", "--merge-output-format", "mp4", url, "-o", video_raw], check=True)
        subprocess.run(["ffmpeg", "-i", video_raw, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_raw, "-y"], check=True, capture_output=True)
        
        # Transcribe di CPU (fp16=False)
        result = model.transcribe(audio_raw, word_timestamps=True, fp16=False, verbose=False)
        global_all_words = []
        full_transcript = ""
        for i, segment in enumerate(result['segments']):
            full_transcript += f"[{segment['start']:.2f} --> {segment['end']:.2f}] {segment['text']}\n"
            if 'words' in segment: global_all_words.extend(segment['words'])
        return {"status": "success", "transcript": full_transcript}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/render_multi")
def render_multi(req: MultiRenderRequest):
    global global_all_words
    try:
        data = req.segments
        if isinstance(data, str): data = json.loads(data)
        video_raw = os.path.join(BASE_DIR, "raw_video.mp4")
        video_info = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_raw]).decode().strip()
        orig_w, orig_h = map(int, video_info.split('x'))

        def render_worker(i, s_dict):
            try:
                idx = i + 1
                start, end = float(s_dict['start']), float(s_dict['end'])
                raw_title = s_dict.get('title', f"short_{idx}")
                clean_title = slugify(raw_title)
                sub_file = os.path.join(BASE_DIR, f"sub_{idx}.ass")
                create_segment_subtitle(global_all_words, start, end, sub_file)
                sub_path = os.path.abspath(sub_file).replace(":", "\\:")
                smart_x, crop_w = get_smart_crop_params(video_raw, start, end, orig_w, orig_h)
                out_filename = f"{clean_title}.mp4"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                
                # FFmpeg menggunakan libx264 (CPU) dan preset medium/fast
                cmd = [
                    "ffmpeg", "-y", "-ss", str(start), "-t", str(end-start),
                    "-i", video_raw,
                    "-vf", f"setpts=PTS-STARTPTS,crop={crop_w}:ih:{smart_x}:0,scale=1080:1920,subtitles='{sub_path}':fontsdir='{FONTS_DIR}/':force_style='Fontname={FONT_NAME},Alignment=2,MarginV=450'",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k", out_path
                ]
                subprocess.run(cmd, check=True)
                if os.path.exists(sub_file): os.remove(sub_file)
                return {"status": "SUCCESS", "filename": out_filename, "url": f"{BASE_DOWNLOAD_URL}/{out_filename}"}
            except Exception as e: return {"status": "error", "message": str(e), "index": i}

        results = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(render_worker, i, s) for i, s in enumerate(data)]
            for future in as_completed(futures): results.append(future.result())
        cleanup_raw_files()
        return {"status": "all_completed", "results": results}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/upload_direct")
async def upload_direct(req: DirectUploadRequest):
    video_path = os.path.join(OUTPUT_DIR, req.filename)
    if not os.path.exists(video_path):
        video_path = os.path.join(PENDING_DIR, req.filename)
    if not os.path.exists(video_path):
        return {"status": "error", "message": "File video tidak ditemukan"}
    try:
        response = core_upload_engine(video_path, req)
        if os.path.exists(video_path): os.remove(video_path)
        meta_file = os.path.join(PENDING_DIR, f"{req.filename}.json")
        if os.path.exists(meta_file): os.remove(meta_file)
        return {"status": "success", "youtube_id": response.get("id")}
    except Exception as e:
        err_msg = str(e)
        if any(x in err_msg for x in ["uploadLimitExceeded", "quotaExceeded", "403"]):
            dest_path = os.path.join(PENDING_DIR, req.filename)
            if video_path != dest_path: os.rename(video_path, dest_path)
            with open(dest_path + ".json", "w") as f: json.dump(req.dict(), f)
            return {"status": "queued", "message": "Limit harian tercapai. Disimpan di antrean."}
        return {"status": "error", "message": err_msg}

@app.get("/process_pending")
def process_pending():
    meta_files = sorted(glob.glob(os.path.join(PENDING_DIR, "*.json")))
    if not meta_files: return {"message": "Antrean kosong"}
    results = []
    quota_available = True
    for meta_path in meta_files:
        if not quota_available: break
        try:
            with open(meta_path, "r") as f: meta_data = json.load(f)
            req_obj = DirectUploadRequest(**meta_data)
            v_path = os.path.join(PENDING_DIR, req_obj.filename)
            core_upload_engine(v_path, req_obj)
            os.remove(v_path)
            os.remove(meta_path)
            results.append(f"{req_obj.filename}: SUCCESS")
            time.sleep(5)
        except Exception as e:
            if any(x in str(e) for x in ["uploadLimitExceeded", "403"]):
                quota_available = False
    return {"status": "done", "results": results}

@app.get("/download_short/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(path) if os.path.exists(path) else {"error": "not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) # Port dibedakan agar bisa jalan berbarengan jika perlu