import os, subprocess, glob, torch, whisper, cv2, json, re, time, sys
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
from fastapi.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()

# --- INISIALISASI ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
OUTPUT_DIR = "/root/vastclip/shorts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# KONFIGURASI FONT
FONT_NAME = "DejaVu Sans"
possible_fonts = ["Poppins-Bold.ttf", "Montserrat-Bold.ttf"]
for f_name in possible_fonts:
    if os.path.exists(os.path.join("/root/vastclip", f_name)):
        FONT_NAME = f_name.split("-")[0]
        break

# Load DNN Face Detector
PROTOTXT = "deploy.prototxt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_FILE)

BASE_DOWNLOAD_URL = ""
global_all_words = [] 

class MultiRenderRequest(BaseModel):
    segments: Any
    original_video_url: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global BASE_DOWNLOAD_URL
    try:
        ip_address = subprocess.check_output(["curl", "-s", "ifconfig.me"]).decode().strip()
        BASE_DOWNLOAD_URL = f"http://{ip_address}:8000/download_short"
        print(f"--- SERVER READY: {BASE_DOWNLOAD_URL} ---")
    except: pass

def slugify(text):
    """Membersihkan judul agar aman jadi nama file"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text

def to_ass_time(sec):
    sec = max(0, sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{int(h):01d}:{int(m):02d}:{int(s):02d}.{int((s-int(s))*100):02d}"

def create_segment_subtitle(words, start_offset, end_offset, out_path):
    header = (
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginV\n"
        f"Style: Default,{FONT_NAME},105,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,1,6,1,2,420\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Text\n"
    )
    
    seg_words = [w for w in words if w['start'] >= (start_offset - 0.2) and w['end'] <= (end_offset + 0.2)]
    
    lines = []
    max_w = 3
    for i in range(0, len(seg_words), max_w):
        chunk = seg_words[i:i+max_w]
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

@app.post("/process")
def process_video(req: dict):
    global global_all_words
    try:
        url = req.get("url")
        print(f"--- START DOWNLOAD: {url} ---", flush=True)
        for f in ["raw_video.mp4", "audio.wav"]:
            if os.path.exists(f): os.remove(f)

        subprocess.run(["yt-dlp", "-f", "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best", "--merge-output-format", "mp4", url, "-o", "raw_video.mp4"], check=True)
        subprocess.run(["ffmpeg", "-i", "raw_video.mp4", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "audio.wav", "-y"], check=True, capture_output=True)

        print("--- START TRANSCRIBE ---", flush=True)
        result = model.transcribe("audio.wav", word_timestamps=True, fp16=True, verbose=False)
        
        global_all_words = []
        full_transcript = ""
        total_segments = len(result['segments'])
        
        for i, segment in enumerate(result['segments']):
            pct = ((i + 1) / total_segments) * 100
            sys.stdout.write(f"\rTranscribing: {pct:.1f}% ({i+1}/{total_segments} segments)")
            sys.stdout.flush()
            
            full_transcript += f"[{segment['start']:.2f} --> {segment['end']:.2f}] {segment['text']}\n"
            if 'words' in segment:
                global_all_words.extend(segment['words'])

        print("\n--- TRANSCRIPTION FINISHED ---", flush=True)
        return {"status": "success", "transcript": full_transcript}
    except Exception as e:
        print(f"\nError in process: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/render_multi")
def render_multi(req: MultiRenderRequest):
    global global_all_words
    data = req.segments
    if isinstance(data, str):
        try: data = json.loads(data)
        except: return {"status": "error", "message": "Invalid JSON"}

    video_info = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", "raw_video.mp4"]).decode().strip()
    orig_w, orig_h = map(int, video_info.split('x'))

    def render_worker(i, s_dict):
        try:
            idx = i + 1
            start, end = float(s_dict['start']), float(s_dict['end'])
            dur = end - start
            
            raw_title = s_dict.get('title', f"short_{idx}")
            clean_title = slugify(raw_title)
            
            sub_file = f"sub_{idx}.ass"
            create_segment_subtitle(global_all_words, start, end, sub_file)
            sub_path = os.path.abspath(sub_file).replace(":", "\\:")

            smart_x, crop_w = get_smart_crop_params("raw_video.mp4", start, end, orig_w, orig_h)
            
            
            out_filename = f"{clean_title}.mp4"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            cmd = [
                "ffmpeg", "-y", "-progress", "pipe:1",
                "-ss", str(start), "-t", str(dur),
                "-i", "raw_video.mp4",
                "-vf", f"setpts=PTS-STARTPTS,crop={crop_w}:ih:{smart_x}:0,scale=1080:1920,subtitles='{sub_path}':fontsdir='/root/vastclip/':force_style='Fontname={FONT_NAME},Alignment=2,MarginV=420'",
                "-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "6M",
                "-c:a", "aac", "-b:a", "192k", out_path
            ]

            print(f"\n>>> RENDERING: {out_filename} ({dur:.2f}s) <<<", flush=True)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

            for line in process.stdout:
                if "out_time_ms" in line:
                    try:
                        time_val = int(line.split('=')[1]) / 1000000
                        pct = min(100, (time_val / dur) * 100)
                        sys.stdout.write(f"\rProgress {out_filename}: {pct:.1f}%")
                        sys.stdout.flush()
                    except: pass
            
            process.wait()
            sys.stdout.write("\n")
            if os.path.exists(sub_file): os.remove(sub_file)
            
            return {"status": "SUCCESS", "filename": out_filename, "url": f"{BASE_DOWNLOAD_URL}/{out_filename}"}
        except Exception as e:
            return {"status": "error", "message": str(e), "index": i}

    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(render_worker, i, s) for i, s in enumerate(data)]
        for future in as_completed(futures):
            results.append(future.result())

    return {"status": "all_completed", "results": results}

@app.get("/download_short/{filename}")
async def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(path) if os.path.exists(path) else {"error": "not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
