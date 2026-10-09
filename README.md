---
title: Clipper
emoji: ✂️
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ✂️ Clipper

Single-user, CPU-only clipping tool: paste a YouTube / Kick / Google Drive link (or upload a file) and get
N vertical 1080×1920 clips with karaoke subtitles, a hook overlay, normalised audio, captions and a zip.
Built for a **free Hugging Face Docker Space** (2 vCPU, 16 GB RAM, no GPU).

| Stage | Tool |
|---|---|
| Download | `yt-dlp` (≤1080p mp4, optional `cookies.txt`), `gdown` for Google Drive, or direct upload |
| Transcription | `faster-whisper` "small", int8, CPU, VAD filter, word timestamps |
| Moment selection | **local** heuristic (speech density + RMS peaks + keywords + ?/! − banned words) or **gemini-2.5-flash** |
| Render | `ffmpeg`: 9:16 crop (mediapipe face-centred), ASS karaoke subtitles, hook text (first 3 s), `loudnorm` −14 LUFS, h264 veryfast CRF 23 |
| Output | `out/<id>/clip_01..N.mp4`, `captions.md`, `clips.zip` |

Long sources are trimmed to the first 90 minutes. Intermediate files (downloads, wav, frames, .ass) are
deleted after every job. Only the last 20 jobs are kept in `out/`.

---

## 1. Create the Hugging Face Space

1. Go to <https://huggingface.co/new-space>.
2. **Space name:** `clipper` (anything you like). **License:** your choice.
3. **SDK:** **Docker** → template **Blank**.
4. **Hardware:** *CPU basic · 2 vCPU · 16 GB* (free).
5. Visibility: **Private** is recommended (the app also has its own password, see below).
6. Click **Create Space**.

## 2. Add Secrets (Settings → Variables and secrets)

| Secret | Required | Purpose |
|---|---|---|
| `PASSWORD` | **yes** | Login password for the UI (username is `admin`, override with a `USERNAME` variable) |
| `GEMINI_API_KEY` | optional | Enables the **gemini** selector (gemini-2.5-flash picks the moments and writes hooks + captions). Without it only **local** is offered. |
| `WHISPER_MODEL` | optional (variable) | Defaults to `small`. `base` is ~2× faster, `medium` is more accurate but slow on CPU. |

Secrets are read at runtime, so you can change them without rebuilding.

## 3. Push the code with git

```bash
git clone https://github.com/gowxs/Clipper.git   # or your copy of this repo
cd Clipper

# add the Space as a second remote (replace <user>)
git remote add space https://huggingface.co/spaces/<user>/clipper
git push space main
```

Hugging Face asks for your HF username and a **write access token** (<https://huggingface.co/settings/tokens>)
as the password. The Space builds the Docker image (≈10 min the first time: it pre-downloads the whisper
model) and then shows the login screen. Every later `git push space main` rebuilds and redeploys.

> `fonts/Montserrat-Bold.ttf` (OFL) and `models/blaze_face_short_range.tflite` are committed so the build
> does not depend on external downloads for them.

## 4. Use it

- **Process** tab: paste a link *or* upload a video, optionally upload `cookies.txt`, pick `local`/`gemini`,
  press **Process**. The log updates live; when done you get previews, per-clip download buttons,
  **Download all (zip)** and `captions.md`.
- **Rules** tab: edit `rules.yaml` (lengths, clip count, banned/boost words, hook style, subtitle font/size/
  colours/position/words per line, caption template with `#Ad`, audience notes) and **Save**. Applies to the next job.
- **History** tab: last 20 jobs with their clips and zip.

### YouTube "Sign in to confirm you're not a bot"

Data-centre IPs (including HF Spaces) are often blocked by YouTube. Export your cookies with a browser
extension such as *Get cookies.txt LOCALLY* (Netscape format) while logged in, and upload the file in the
**cookies.txt** field. Prefer a throw-away Google account. Kick and Google Drive links usually work without cookies.

### Google Drive

Share the file as *Anyone with the link → Viewer* and paste the `https://drive.google.com/file/d/<id>/view` link.

---

## 5. Test locally with Docker

```bash
docker build -t clipper .

docker run --rm -p 7860:7860 \
  -e PASSWORD=test123 \
  -e GEMINI_API_KEY=...   # optional
  -v "$PWD/out:/home/user/app/out" \
  clipper
```

Open <http://localhost:7860>, log in with `admin` / `test123`, upload a short video with speech and press
**Process**. Clips appear in `./out/<job-id>/`.

### Without Docker (venv)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg must be on PATH; deno or node is needed for YouTube downloads

# UI
PASSWORD=test123 python app.py

# or the CLI (same pipeline, prints JSON with the produced clips)
python pipeline.py path/to/video.mp4
python pipeline.py "https://www.youtube.com/watch?v=..." --cookies cookies.txt --mode gemini
```

### Local test run (what to expect)

A 2.9-minute 720p test video with synthetic speech, default rules (4 clips, 15–30 s), on 4 CPU threads:

```
[19:39:05] Job 20261009-193905-5900ed started (mode=local, clips=4, 15-30s)
[19:39:06] Source: 1280x720, 2.9 min
[19:39:24] Transcribing with faster-whisper 'small' (int8, CPU, VAD) ...
[19:39:51] Transcription done: 393 words, language=en
[19:39:51] Selected windows: 9.7-38.0s (score 2.58), 43.9-67.0s (score 2.51), 95.4-123.8s (score 2.55), 138.5-161.4s (score 2.36)
[19:39:53]   clip 1: face not found, centre crop
[19:39:53]   rendering clip_01.mp4 (9.7-38.0s) ...
...
[19:40:30] Finished: 4 clips in out/20261009-193905-5900ed (1.4 min)

$ ls out/20261009-193905-5900ed
captions.md  clip_01.mp4  clip_02.mp4  clip_03.mp4  clip_04.mp4  clips.zip
```

Each clip: `h264 1080x1920 30 fps`, integrated loudness ≈ −14.5 LUFS. On the free 2 vCPU Space expect
roughly 1× real time for transcription of the source plus ~10–20 s of rendering per clip.

---

## Files

```
app.py            Gradio UI (tabs: Process / Rules / History), background job + live log
pipeline.py       download → transcribe → select → render → zip (also a CLI)
rules.yaml        editable rules (see the Rules tab)
requirements.txt  Python deps
Dockerfile        python:3.11-slim + ffmpeg + fonts + deno + pre-downloaded whisper model
fonts/            Montserrat-Bold.ttf (SIL OFL)
models/           mediapipe BlazeFace short-range model (Apache-2.0)
out/              results (created at runtime)
```

## Notes & limits

- One job at a time (single-user tool). A second **Process** click while a job runs is refused.
- Free Spaces have no persistent disk: `out/` is wiped on restart/rebuild. Download your zips.
- Free Spaces sleep after ~48 h without traffic; the first request wakes them (≈1 min).
- Uploads are limited to 4 GB; sources longer than 90 min are cut to 90 min.
- `clips_per_video` is capped at 8 in the UI.
