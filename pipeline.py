"""
Clipper pipeline: download -> transcribe -> select moments -> render vertical clips.

Designed for a free Hugging Face Space (2 vCPU, 16 GB RAM, no GPU).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("CLIPPER_OUT", BASE_DIR / "out"))
WORK_DIR = Path(os.environ.get("CLIPPER_WORK", BASE_DIR / "work"))
FONTS_DIR = BASE_DIR / "fonts"
MODELS_DIR = BASE_DIR / "models"
RULES_PATH = BASE_DIR / "rules.yaml"
HISTORY_PATH = OUT_DIR / "history.json"

DEFAULT_MAX_VIDEO_MINUTES = 60        # long videos are trimmed (rules.yaml: max_video_minutes)
ALLOWED_WHISPER_MODELS = ("tiny", "base", "small")
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"

DEFAULT_RULES: dict = {
    "min_len": 15,
    "max_len": 30,
    "clips_per_video": 4,
    "whisper_model": "base",
    "max_video_minutes": 60,
    "banned_words": [],
    "keywords_boost": [],
    "hook_style": "text_overlay",
    "subtitle": {
        "font": "Montserrat-Bold",
        "size": 72,
        "color": "#FFFFFF",
        "highlight_color": "#FFD400",
        "position": "lower_third",
        "words_per_line": 3,
    },
    "caption_template": "{hook}\n\n{title} #Ad\n\n{hashtags}\n",
    "audience_notes": "",
    "logos": {
        "my_logo": {"enabled": False, "file": "my_logo.png", "position": "bottom_right", "width_pct": 12,
                    "opacity": 0.85, "margin_px": 40, "show": "always"},
        "campaign_logo": {"enabled": False, "file": "campaign_logo.png", "position": "top_left", "width_pct": 12,
                          "opacity": 0.85, "margin_px": 40, "show": "always"},
    },
}

LOGOS_DIR = BASE_DIR / "logos"
LOGO_SLOTS = ("my_logo", "campaign_logo")
LOGO_POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
LOGO_SHOW = ("always", "first_3s", "last_3s")
# Safe zones on the 1080x1920 canvas: hook text lives in the upper third (y >= 230, centred),
# subtitles in the lower third (y <= ~1440). Logos stay in the corners and must not enter those bands.
LOGO_TOP_MAX_Y = 200        # bottom edge of a top-corner logo must stay above this
LOGO_BOTTOM_MIN_Y = 1500    # top edge of a bottom-corner logo must stay below this

DEFAULT_RULES_TEXT = """# Clipper rules — edit in the "Rules" tab or directly in this file.

# Clip length bounds (seconds)
min_len: 15
max_len: 30

# How many clips to produce per video (UI caps this at 8)
clips_per_video: 4

# faster-whisper model: base (fast, ~0.3x realtime on 2 vCPU) | small (more accurate, ~1x realtime)
whisper_model: base

# Sources longer than this are trimmed to the first N minutes (keeps Cloud Run time/cost bounded)
max_video_minutes: 60

# Windows containing these words are penalised (case-insensitive)
banned_words: []

# Windows containing these words get a score boost (case-insensitive)
keywords_boost: []

# "text_overlay" draws the hook in the top third for the first 3 s, "none" disables it
hook_style: text_overlay

subtitle:
  font: Montserrat-Bold
  size: 72
  color: "#FFFFFF"
  highlight_color: "#FFD400"
  position: lower_third      # lower_third | center | top
  words_per_line: 3

# Placeholders: {hook} {title} {hashtags} {start} {end}
caption_template: |
  {hook}

  {title} #Ad

  {hashtags}

# Free-form notes used by the Gemini selector (ignored by the local selector)
audience_notes: |
  Short-form audience (TikTok / Reels / Shorts). Prefer moments with a strong
  opening line, a clear point, humour or surprise. Avoid slow or context-dependent parts.

# Logo overlays (PNG with transparency). `file` is a name inside logos/ (local folder or the GCS bucket)
# or an absolute path. Logos sit in the corners and never enter the hook band (upper third, centre)
# or the subtitle band (lower third): top logos are capped at y<200, bottom logos start below y>1500.
#   position: top_left | top_right | bottom_left | bottom_right
#   width_pct: width as % of the 1080 px frame (3–30)
#   show: always | first_3s | last_3s
logos:
  my_logo:
    enabled: false
    file: my_logo.png
    position: bottom_right
    width_pct: 12
    opacity: 0.85
    margin_px: 40
    show: always
  campaign_logo:
    enabled: false
    file: campaign_logo.png
    position: top_left
    width_pct: 12
    opacity: 0.85
    margin_px: 40
    show: always
"""

_history_lock = threading.Lock()
_whisper_lock = threading.Lock()
_whisper_models: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_rules(path: Path = RULES_PATH) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    return _deep_merge(DEFAULT_RULES, data)


def read_rules_text(path: Path = RULES_PATH) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_RULES_TEXT


def save_rules_text(text: str, path: Path = RULES_PATH) -> dict:
    """Validate YAML, write to disk, return merged rules. Raises on invalid YAML."""
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("rules.yaml must be a mapping")
    merged = _deep_merge(DEFAULT_RULES, data)
    if merged["min_len"] <= 0 or merged["max_len"] <= merged["min_len"]:
        raise ValueError("min_len must be > 0 and max_len > min_len")
    if int(merged["clips_per_video"]) < 1:
        raise ValueError("clips_per_video must be >= 1")
    if str(merged.get("whisper_model", "base")) not in ALLOWED_WHISPER_MODELS:
        raise ValueError(f"whisper_model must be one of {', '.join(ALLOWED_WHISPER_MODELS)}")
    if float(merged.get("max_video_minutes", 60)) <= 0:
        raise ValueError("max_video_minutes must be > 0")
    for slot in LOGO_SLOTS:
        lg = (merged.get("logos") or {}).get(slot) or {}
        if lg.get("position", "bottom_right") not in LOGO_POSITIONS:
            raise ValueError(f"logos.{slot}.position must be one of {', '.join(LOGO_POSITIONS)}")
        if lg.get("show", "always") not in LOGO_SHOW:
            raise ValueError(f"logos.{slot}.show must be one of {', '.join(LOGO_SHOW)}")
        if not (3 <= float(lg.get("width_pct", 12)) <= 30):
            raise ValueError(f"logos.{slot}.width_pct must be between 3 and 30")
        if not (0 < float(lg.get("opacity", 0.85)) <= 1):
            raise ValueError(f"logos.{slot}.opacity must be in (0, 1]")
    used = [lg.get("position") for lg in (merged.get("logos") or {}).values() if lg and lg.get("enabled")]
    if len(used) != len(set(used)):
        raise ValueError("enabled logos must use different corners")
    path.write_text(text, encoding="utf-8")
    return merged


# --------------------------------------------------------------------------- #
# Job state
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    status: str = "queued"           # queued | running | done | error
    progress: float = 0.0
    log_lines: list[str] = field(default_factory=list)
    title: str = ""
    source: str = ""
    clips: list[dict] = field(default_factory=list)
    zip_path: Optional[str] = None
    captions_path: Optional[str] = None
    error: Optional[str] = None
    gcs: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log(self, msg: str, progress: Optional[float] = None):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self.log_lines.append(line)
            if progress is not None:
                self.progress = max(self.progress, float(progress))
        print(line, flush=True)

    @property
    def log_text(self) -> str:
        with self._lock:
            return "\n".join(self.log_lines[-400:])

    @property
    def out_dir(self) -> Path:
        return OUT_DIR / self.id


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(cmd: list[str], log: Optional[Callable[[str], None]] = None, timeout: Optional[int] = None) -> str:
    """Run a command, raise RuntimeError with stderr tail on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd[:3])} ...\n{tail}")
    return proc.stdout


def ffprobe_json(path: str) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(out)


def probe_video(path: str) -> dict:
    info = ffprobe_json(path)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if v is None:
        raise RuntimeError("No video stream found in input")
    if a is None:
        raise RuntimeError("No audio stream found in input (speech is required)")
    width, height = int(v["width"]), int(v["height"])
    # honour rotation metadata
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(abs(float(sd["rotation"])))
    if str(v.get("tags", {}).get("rotate", "0")) in ("90", "270"):
        rot = 90
    if rot in (90, 270):
        width, height = height, width
    duration = float(info["format"].get("duration") or v.get("duration") or 0)
    return {"width": width, "height": height, "duration": duration}


def sanitize_filename(name: str, limit: int = 80) -> str:
    name = re.sub(r"[^\w\s\-\.]", "", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:limit] or "video"


def is_google_drive(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def js_runtime_args() -> list[str]:
    """yt-dlp needs a JS runtime for YouTube; prefer deno, fall back to node."""
    for rt in ("deno", "node"):
        if shutil.which(rt):
            return ["--js-runtimes", rt]
    return []


# --------------------------------------------------------------------------- #
# 1) Download
# --------------------------------------------------------------------------- #
def download_source(url: str, workdir: Path, cookies: Optional[str], job: Job) -> tuple[str, str]:
    """Return (local_path, title)."""
    workdir.mkdir(parents=True, exist_ok=True)
    url = url.strip()
    if url.startswith("gs://"):
        import gcs

        job.log(f"Downloading from Cloud Storage: {url}")
        path = gcs.download_gs_uri(url, workdir / "gcs")
        return str(path), path.stem
    if is_google_drive(url):
        import gdown

        job.log("Downloading from Google Drive with gdown ...")
        target_dir = workdir / "gdrive"
        target_dir.mkdir(exist_ok=True)
        out = gdown.download(url=url, output=str(target_dir) + "/", quiet=True, fuzzy=True)
        if not out:
            raise RuntimeError("gdown could not download the file (is the link shared publicly?)")
        path = Path(out)
        if path.is_dir():
            files = sorted(path.iterdir(), key=lambda p: p.stat().st_size, reverse=True)
            if not files:
                raise RuntimeError("gdown downloaded an empty folder")
            path = files[0]
        return str(path), path.stem

    # yt-dlp (YouTube, Kick, ...)
    out_tmpl = str(workdir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "--retries", "5",
        "--fragment-retries", "10",
        "--concurrent-fragments", "4",
        "--print-to-file", "%(title)s", str(workdir / "title.txt"),
        "-o", out_tmpl,
    ]
    cmd += js_runtime_args()
    if cookies:
        cmd += ["--cookies", cookies]
    cmd.append(url)
    job.log(f"Downloading with yt-dlp (<=1080p){' + cookies' if cookies else ''} ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = ""
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.strip()
        if not line:
            continue
        if line.startswith("[download]") and "%" in line:
            if time.time() - float(getattr(job, "_last_dl_log", 0)) > 3:
                job.log(line)
                job._last_dl_log = time.time()  # type: ignore[attr-defined]
        elif not line.startswith("[download]"):
            last = line
            job.log(line)
    proc.wait()
    if proc.returncode != 0:
        hint = ""
        if "Sign in to confirm" in last or "bot" in last.lower():
            hint = " — YouTube asks for login: upload a cookies.txt exported from your browser."
        raise RuntimeError(f"yt-dlp failed: {last}{hint}")
    candidates = sorted(workdir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise RuntimeError("yt-dlp finished but no file was produced")
    title_file = workdir / "title.txt"
    title = title_file.read_text(encoding="utf-8").strip().splitlines()[0] if title_file.exists() else candidates[0].stem
    return str(candidates[0]), title or "video"


def limit_duration(src: str, workdir: Path, job: Job, max_minutes: float = DEFAULT_MAX_VIDEO_MINUTES) -> str:
    info = probe_video(src)
    max_seconds = float(max_minutes) * 60
    if info["duration"] > max_seconds + 1:
        job.log(f"Source is {info['duration']/60:.1f} min — trimming to the first {max_minutes:g} min.")
        trimmed = str(workdir / "source_trimmed.mp4")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-t", str(max_seconds),
             "-c", "copy", "-movflags", "+faststart", trimmed])
        return trimmed
    return src


# --------------------------------------------------------------------------- #
# 2) Transcription
# --------------------------------------------------------------------------- #
def extract_audio(src: str, wav_path: str):
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", wav_path])


def load_wav_mono16k(wav_path: str) -> np.ndarray:
    with wave.open(wav_path, "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def get_whisper(name: str = DEFAULT_WHISPER_MODEL):
    name = name if name in ALLOWED_WHISPER_MODELS else DEFAULT_WHISPER_MODEL
    with _whisper_lock:
        if name not in _whisper_models:
            from faster_whisper import WhisperModel

            threads = max(1, min(4, os.cpu_count() or 2))
            # keep only one model resident (4 GiB Cloud Run instance)
            _whisper_models.clear()
            _whisper_models[name] = WhisperModel(name, device="cpu", compute_type="int8",
                                                 cpu_threads=threads, num_workers=1)
        return _whisper_models[name]


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


def transcribe(wav_path: str, job: Job, total_dur: float,
               model_name: str = DEFAULT_WHISPER_MODEL) -> tuple[list[Word], list[dict], str]:
    """Return (words, segments, language)."""
    model = get_whisper(model_name)
    job.log(f"Transcribing with faster-whisper '{model_name}' (int8, CPU, VAD) ...")
    # decode with ffmpeg (already done) and hand whisper a float32 array -> no PyAV version issues
    audio = load_wav_mono16k(wav_path)
    segments_iter, info = model.transcribe(
        audio,
        beam_size=1,
        best_of=1,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200},
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    segments: list[dict] = []
    last_report = time.time()
    for seg in segments_iter:
        seg_words = []
        for w in seg.words or []:
            t = (w.word or "").strip()
            if not t:
                continue
            wd = Word(float(w.start), float(w.end), t, float(w.probability or 1.0))
            words.append(wd)
            seg_words.append(wd)
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(),
                         "words": seg_words})
        if time.time() - last_report > 5:
            frac = min(0.99, float(seg.end) / max(total_dur, 1.0))
            job.log(f"  transcribed {seg.end/60:.1f}/{total_dur/60:.1f} min", progress=0.15 + 0.35 * frac)
            last_report = time.time()
    job.log(f"Transcription done: {len(words)} words, language={info.language}", progress=0.5)
    return words, segments, info.language or "en"


# --------------------------------------------------------------------------- #
# 3) Moment selection
# --------------------------------------------------------------------------- #
def audio_rms_curve(wav_path: str, hop: float = 0.1) -> tuple[np.ndarray, float]:
    """RMS energy per `hop` seconds, normalised to 0..1."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = max(1, int(sr * hop))
    nwin = len(data) // win
    if nwin == 0:
        return np.zeros(1), hop
    frames = data[: nwin * win].reshape(nwin, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    rms_db = 20 * np.log10(rms + 1e-6)
    lo, hi = np.percentile(rms_db, 5), np.percentile(rms_db, 99)
    norm = np.clip((rms_db - lo) / max(hi - lo, 1e-6), 0, 1)
    return norm, hop


_SENT_END = re.compile(r"[.!?…]+[\"'”’)]*$")


def sentence_start_indices(words: list[Word], segments: list[dict]) -> list[int]:
    """Indices of words that start a sentence."""
    starts = set()
    if words:
        starts.add(0)
    for i in range(1, len(words)):
        prev = words[i - 1]
        if _SENT_END.search(prev.text) or (words[i].start - prev.end) > 0.8:
            starts.add(i)
    # segment starts count as sentence starts too
    seg_start_times = {round(s["start"], 2) for s in segments}
    for i, w in enumerate(words):
        if round(w.start, 2) in seg_start_times:
            starts.add(i)
    return sorted(starts)


def _norm(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def select_moments_local(words: list[Word], segments: list[dict], rms: np.ndarray, hop: float,
                         rules: dict, total_dur: float, job: Job) -> list[dict]:
    min_len = float(rules["min_len"])
    max_len = float(rules["max_len"])
    n_clips = int(rules["clips_per_video"])
    banned = [b.lower() for b in rules.get("banned_words") or []]
    boost = [b.lower() for b in rules.get("keywords_boost") or []]

    if not words:
        job.log("No speech found — falling back to evenly spaced windows.")
        return fallback_windows(total_dur, min_len, max_len, n_clips)

    starts = sentence_start_indices(words, segments)
    cands = []
    for si in starts:
        s_word = words[si]
        start_t = s_word.start
        # choose end: prefer a sentence end within [min_len, max_len]; else last word that fits
        best_end_i = None
        fallback_end_i = None
        for j in range(si, len(words)):
            end_t = words[j].end
            if end_t - start_t > max_len:
                break
            if end_t - start_t >= min_len:
                fallback_end_i = j
                if _SENT_END.search(words[j].text):
                    best_end_i = j
                    # keep extending to the latest sentence end within max_len
        end_i = best_end_i if best_end_i is not None else fallback_end_i
        if end_i is None:
            continue
        end_t = words[end_i].end
        seg_words = words[si:end_i + 1]
        dur = end_t - start_t
        text_l = " ".join(w.text for w in seg_words).lower()

        density = len(seg_words) / dur
        i0, i1 = int(start_t / hop), max(int(start_t / hop) + 1, int(end_t / hop))
        r = rms[i0:i1] if i1 <= len(rms) else rms[i0:]
        if len(r) == 0:
            peak = 0.0
        else:
            k = max(1, int(len(r) * 0.1))
            peak = float(np.mean(np.sort(r)[-k:]))
        punct = sum(1 for w in seg_words if "?" in w.text or "!" in w.text)
        boost_hits = sum(text_l.count(k) for k in boost) if boost else 0
        banned_hits = sum(text_l.count(k) for k in banned) if banned else 0
        cands.append({
            "si": si, "ei": end_i, "start": start_t, "end": end_t,
            "density": density, "peak": peak, "punct": punct,
            "boost": boost_hits, "banned": banned_hits, "text": " ".join(w.text for w in seg_words),
        })

    if not cands:
        job.log("No candidate windows matched the length rules — using fallback windows.")
        return fallback_windows(total_dur, min_len, max_len, n_clips)

    density = _norm(np.array([c["density"] for c in cands]))
    peak = _norm(np.array([c["peak"] for c in cands]))
    for c, d, p in zip(cands, density, peak):
        score = 1.0 * d + 1.0 * p + 0.25 * min(c["punct"], 4) + 0.6 * min(c["boost"], 3) - 2.0 * c["banned"]
        # slight bonus for ending on a sentence boundary
        if _SENT_END.search(words[c["ei"]].text):
            score += 0.2
        c["score"] = float(score)

    cands.sort(key=lambda c: c["score"], reverse=True)
    chosen: list[dict] = []
    for c in cands:
        if any(not (c["end"] <= o["start"] or c["start"] >= o["end"]) for o in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= n_clips:
            break
    chosen.sort(key=lambda c: c["start"])

    results = []
    for c in chosen:
        # no silence in the first 0.5 s: start a hair before the first word
        lead = 0.15
        prev_end = words[c["si"] - 1].end if c["si"] > 0 else 0.0
        start = max(prev_end, c["start"] - lead, 0.0)
        end = min(total_dur, c["end"] + 0.25)
        clip_words = [Word(w.start - start, w.end - start, w.text, w.prob) for w in words[c["si"]:c["ei"] + 1]]
        results.append({
            "start": start, "end": end, "words": clip_words,
            "text": c["text"], "score": c["score"],
            "hook": make_hook(c["text"]), "caption": None,
        })
    job.log("Selected windows: " + ", ".join(f"{r['start']:.1f}-{r['end']:.1f}s (score {r['score']:.2f})" for r in results))
    return results


def fallback_windows(total_dur: float, min_len: float, max_len: float, n: int) -> list[dict]:
    length = min(max_len, max(min_len, total_dur / max(n, 1) * 0.6))
    if total_dur <= length:
        return [{"start": 0.0, "end": total_dur, "words": [], "text": "", "score": 0.0, "hook": "", "caption": None}]
    step = (total_dur - length) / max(n - 1, 1)
    return [{"start": i * step, "end": i * step + length, "words": [], "text": "", "score": 0.0,
             "hook": "", "caption": None} for i in range(n)]


def make_hook(text: str, max_words: int = 7) -> str:
    toks = [t for t in re.split(r"\s+", text.strip()) if t]
    # prefer the first full sentence when it is short enough
    for i, t in enumerate(toks[:max_words]):
        if _SENT_END.search(t) and i >= 2:
            toks = toks[: i + 1]
            break
    hook = " ".join(toks[:max_words]).strip()
    hook = re.sub(r"[,;:\-–—]+$", "", hook).strip()
    if hook and hook[-1] not in ".!?…":
        hook = hook.rstrip(".") + ("…" if len(toks) > max_words else "")
    return hook[:1].upper() + hook[1:]


def attach_clip_words(clip: dict, words: list[Word]):
    """For Gemini-selected windows: collect the words inside [start, end]."""
    clip["words"] = [Word(w.start - clip["start"], w.end - clip["start"], w.text, w.prob)
                     for w in words if w.start >= clip["start"] - 0.05 and w.end <= clip["end"] + 0.05]
    clip["text"] = " ".join(w.text for w in clip["words"])


def use_vertex(api_key: str) -> bool:
    """Vertex AI express-mode keys start with 'AQ.'; GEMINI_USE_VERTEX=1 forces it."""
    flag = os.environ.get("GEMINI_USE_VERTEX", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    return api_key.startswith("AQ.")


def make_genai_client(api_key: str):
    from google import genai

    if use_vertex(api_key):
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(api_key=api_key)


def select_moments_gemini(words: list[Word], segments: list[dict], rules: dict, total_dur: float,
                          title: str, job: Job) -> list[dict]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    n_clips = int(rules["clips_per_video"])
    min_len, max_len = float(rules["min_len"]), float(rules["max_len"])

    lines = []
    for s in segments:
        lines.append(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
    transcript = "\n".join(lines)
    if len(transcript) > 400_000:
        transcript = transcript[:400_000]

    prompt = f"""You are an expert short-form video editor. Below is a timestamped transcript of a video titled "{title}".
Pick the {n_clips} best self-contained moments for vertical short clips (TikTok / Reels / Shorts).

Rules:
- Each clip must be between {min_len:.0f} and {max_len:.0f} seconds long.
- Start exactly at the beginning of a sentence; end at the end of a sentence.
- Clips must not overlap.
- Avoid these words/topics: {', '.join(rules.get('banned_words') or []) or 'none'}.
- Prefer moments mentioning: {', '.join(rules.get('keywords_boost') or []) or 'anything engaging'}.
- Audience notes: {rules.get('audience_notes') or 'general'}.

For each clip write:
- "hook": a punchy on-screen hook of at most 7 words (same language as the transcript),
- "caption": a social media caption (1-3 short lines) in the same language, including 3-6 relevant hashtags. Do NOT include "#Ad" (it is added automatically).

Return ONLY a JSON array like:
[{{"start": 12.3, "end": 38.0, "hook": "...", "caption": "..."}}]

Transcript:
{transcript}
"""
    client = make_genai_client(api_key)
    job.log(f"Asking gemini-2.5-flash ({'Vertex AI' if use_vertex(api_key) else 'Gemini API'}) to pick the best moments ...")
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.4},
    )
    raw = (resp.text or "").strip()
    m = re.search(r"\[.*\]", raw, flags=re.S)
    if not m:
        raise RuntimeError(f"Gemini returned no JSON array: {raw[:200]}")
    items = json.loads(m.group(0))

    results = []
    for it in items:
        try:
            start, end = float(it["start"]), float(it["end"])
        except Exception:
            continue
        start = max(0.0, min(start, total_dur))
        end = max(start, min(end, total_dur))
        # snap to nearest word boundaries
        if words:
            ws = min(words, key=lambda w: abs(w.start - start))
            we = min(words, key=lambda w: abs(w.end - end))
            start, end = ws.start - 0.15, we.end + 0.25
            start = max(0.0, start)
            end = min(total_dur, end)
        dur = end - start
        if dur < min_len * 0.8 or dur > max_len * 1.2:
            job.log(f"  skipping Gemini window {start:.1f}-{end:.1f}s (length {dur:.1f}s outside rules)")
            continue
        if dur > max_len:
            end = start + max_len
        if any(not (end <= o["start"] or start >= o["end"]) for o in results):
            continue
        clip = {"start": start, "end": end, "hook": str(it.get("hook", "")).strip()[:80],
                "caption": str(it.get("caption", "")).strip(), "score": 0.0}
        attach_clip_words(clip, words)
        if not clip["hook"]:
            clip["hook"] = make_hook(clip["text"])
        results.append(clip)
    results.sort(key=lambda c: c["start"])
    results = results[:n_clips]
    if not results:
        raise RuntimeError("Gemini produced no usable windows")
    job.log("Gemini selected: " + ", ".join(f"{r['start']:.1f}-{r['end']:.1f}s" for r in results))
    return results


# --------------------------------------------------------------------------- #
# 4) Render
# --------------------------------------------------------------------------- #
def _hex_to_ass(color: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> '&HAABBGGRR'"""
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        c = "FFFFFF"
    r, g, b = c[0:2], c[2:4], c[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def build_ass(clip: dict, rules: dict, out_path: str, hook: str):
    sub = rules["subtitle"]
    font = str(sub.get("font", "Montserrat-Bold"))
    size = int(sub.get("size", 72))
    color = _hex_to_ass(str(sub.get("color", "#FFFFFF")))
    hi = _hex_to_ass(str(sub.get("highlight_color", "#FFD400")))
    wpl = max(1, int(sub.get("words_per_line", 3)))
    pos = str(sub.get("position", "lower_third"))
    alignment, margin_v = {"lower_third": (2, 480), "center": (5, 0), "top": (8, 300)}.get(pos, (2, 480))
    duration = clip["end"] - clip["start"]

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{font},{size},{color},{hi},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,{alignment},60,60,{margin_v},1
Style: Hook,{font},{int(size * 0.95)},&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,3,6,0,8,80,80,230,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    words: list[Word] = clip.get("words") or []
    # group into lines of `wpl` words, restart a line at sentence ends
    lines: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        cur.append(w)
        if len(cur) >= wpl or _SENT_END.search(w.text):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)

    for li, line in enumerate(lines):
        line_start = line[0].start
        line_end = line[-1].end
        next_line_start = lines[li + 1][0].start if li + 1 < len(lines) else duration
        line_end = min(max(line_end, line_start + 0.3), next_line_start, duration)
        for wi, w in enumerate(line):
            ev_start = w.start if wi > 0 else line_start
            ev_end = line[wi + 1].start if wi + 1 < len(line) else line_end
            if ev_end <= ev_start:
                ev_end = ev_start + 0.05
            parts = []
            for wj, w2 in enumerate(line):
                t = _ass_escape(w2.text)
                if wj == wi:
                    parts.append(f"{{\\c{hi}\\fscx108\\fscy108}}{t}{{\\c{color}\\fscx100\\fscy100}}")
                else:
                    parts.append(t)
            text = " ".join(parts)
            events.append(f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},Sub,,0,0,0,,{text}")

    if hook and rules.get("hook_style", "text_overlay") == "text_overlay":
        hook_text = _ass_escape(hook.upper())
        events.append(
            f"Dialogue: 1,{_ass_time(0)},{_ass_time(min(3.0, duration))},Hook,,0,0,0,,"
            f"{{\\fad(150,250)}}{hook_text}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))
        f.write("\n")


_face_detector = None
_face_lock = threading.Lock()


def ensure_face_model() -> Optional[Path]:
    if FACE_MODEL_PATH.exists():
        return FACE_MODEL_PATH
    try:
        import urllib.request

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        return FACE_MODEL_PATH
    except Exception:
        return None


def get_face_detector():
    global _face_detector
    with _face_lock:
        if _face_detector is None:
            model = ensure_face_model()
            if model is None:
                return None
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                opts = vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model)),
                    min_detection_confidence=0.45,
                )
                _face_detector = (mp, vision.FaceDetector.create_from_options(opts))
            except Exception as e:  # pragma: no cover
                print(f"[face] mediapipe unavailable: {e}", flush=True)
                _face_detector = False
        return _face_detector or None


def _detect_faces_tiled(mp, detector, frame_path: str) -> list[tuple[float, float]]:
    """
    Run BlazeFace (short range, 128x128 input) on the full frame and on square tiles so that
    faces that are small relative to a wide frame are still found.
    Returns [(x_center_norm, area_norm)].
    """
    import cv2

    img = cv2.imread(frame_path)
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    found: list[tuple[float, float]] = []

    def _run(tile, x_off, tw):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(tile))
        res = detector.detect(mp_img)
        for d in res.detections or []:
            bb = d.bounding_box
            cx = (x_off + bb.origin_x + bb.width / 2) / w
            area = (bb.width * bb.height) / (w * h)
            found.append((float(cx), float(area)))

    _run(img, 0, w)
    if not found and w > h * 1.2:
        step = max(1, (w - h) // 2)
        for x in range(0, w - h + 1, step):
            _run(img[:, x:x + h], x, h)
    return found


def detect_face_center(src: str, start: float, end: float, workdir: Path, n_frames: int = 6) -> Optional[float]:
    """Return normalised x-centre (0..1) of the dominant face across sampled frames, or None."""
    det = get_face_detector()
    if det is None:
        return None
    mp, detector = det
    frames_dir = workdir / f"frames_{int(start * 1000)}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    centers = []
    try:
        for i in range(n_frames):
            t = start + (end - start) * (i + 0.5) / n_frames
            frame = frames_dir / f"f{i}.jpg"
            try:
                run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1",
                     "-vf", "scale='min(1280,iw)':-2", "-q:v", "4", str(frame)], timeout=60)
            except Exception:
                continue
            try:
                faces = _detect_faces_tiled(mp, detector, str(frame))
            except Exception:
                continue
            if not faces:
                continue
            cx, _ = max(faces, key=lambda f: f[1])
            centers.append(cx)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
    if len(centers) < 2:
        return None
    return float(np.median(centers))


# --------------------------------------------------------------------------- #
# Logos
# --------------------------------------------------------------------------- #
def list_saved_logos() -> list[str]:
    """Names of PNG logos available in logos/ (local) plus the GCS logo folder."""
    names = set()
    if LOGOS_DIR.exists():
        names.update(p.name for p in LOGOS_DIR.iterdir() if p.suffix.lower() in (".png", ".webp"))
    try:
        import gcs

        names.update(gcs.list_logos())
    except Exception:
        pass
    return sorted(names)


def save_logo(src_path: str, name: Optional[str] = None) -> str:
    """Copy an uploaded logo into logos/ (and the bucket). Returns the stored name."""
    src = Path(src_path)
    name = sanitize_filename(name or src.stem) + (src.suffix.lower() if src.suffix.lower() in (".png", ".webp") else ".png")
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    dst = LOGOS_DIR / name
    if src.resolve() != dst.resolve():
        shutil.copy(src, dst)
    try:
        import gcs

        gcs.upload_logo(dst, name)
    except Exception as e:
        print(f"[logos] GCS upload skipped: {e}", flush=True)
    return name


def resolve_logo(file_or_name: str) -> Optional[Path]:
    """Absolute path → as is; otherwise look in logos/ and then in the bucket (cached locally)."""
    if not file_or_name:
        return None
    p = Path(file_or_name)
    if p.is_absolute() and p.exists():
        return p
    local = LOGOS_DIR / p.name
    if local.exists():
        return local
    try:
        import gcs

        if gcs.enabled() and gcs.download_logo(p.name, local):
            return local
    except Exception:
        pass
    return None


def active_logos(rules: dict, overrides: Optional[dict] = None, log=None) -> list[dict]:
    """
    Merge rules.logos with per-job overrides {slot: path|None}. An override path enables the slot for this
    job; an override of "" / None disables it. Returns [{slot, path, position, width_pct, opacity, margin_px, show}].
    """
    overrides = overrides or {}
    out = []
    for slot in LOGO_SLOTS:
        cfg = dict(DEFAULT_RULES["logos"][slot])
        cfg.update((rules.get("logos") or {}).get(slot) or {})
        if slot in overrides:
            if not overrides[slot]:
                continue
            path = resolve_logo(str(overrides[slot]))
        else:
            if not cfg.get("enabled"):
                continue
            path = resolve_logo(str(cfg.get("file", "")))
        if path is None:
            if log:
                log(f"  logo '{slot}': file '{overrides.get(slot) or cfg.get('file')}' not found — skipped")
            continue
        out.append({"slot": slot, "path": str(path), "position": cfg.get("position", "bottom_right"),
                    "width_pct": float(cfg.get("width_pct", 12)), "opacity": float(cfg.get("opacity", 0.85)),
                    "margin_px": int(cfg.get("margin_px", 40)), "show": cfg.get("show", "always")})
    # keep corners unique (first wins)
    seen = set()
    uniq = []
    for lg in out:
        if lg["position"] in seen:
            if log:
                log(f"  logo '{lg['slot']}': corner {lg['position']} already used — skipped")
            continue
        seen.add(lg["position"])
        uniq.append(lg)
    return uniq


def logo_filter_chain(logos: list[dict], first_input_index: int, dur: float, in_label: str = "base") -> tuple[str, str]:
    """
    Build the filter_complex fragment that scales each logo into its safe corner and overlays it on
    [in_label]. Returns (filter_text, final_label).
    """
    parts = []
    cur = in_label
    for i, lg in enumerate(logos):
        idx = first_input_index + i
        pos = lg["position"] if lg["position"] in LOGO_POSITIONS else "bottom_right"
        m = max(0, int(lg["margin_px"]))
        width = int(1080 * min(max(lg["width_pct"], 3.0), 30.0) / 100)
        # vertical cap so the logo never reaches the hook / subtitle bands
        if pos.startswith("top"):
            max_h = max(24, LOGO_TOP_MAX_Y - m)
        else:
            max_h = max(24, 1920 - m - LOGO_BOTTOM_MIN_Y)
        alpha = min(max(lg["opacity"], 0.05), 1.0)
        lbl = f"lg{i}"
        parts.append(f"[{idx}:v]format=rgba,scale=w={width}:h={max_h}:force_original_aspect_ratio=decrease,"
                     f"colorchannelmixer=aa={alpha:.3f}[{lbl}]")
        x = f"{m}" if pos.endswith("left") else f"main_w-overlay_w-{m}"
        y = f"{m}" if pos.startswith("top") else f"main_h-overlay_h-{m}"
        enable = ""
        if lg["show"] == "first_3s":
            enable = ":enable='lte(t,3)'"
        elif lg["show"] == "last_3s":
            enable = f":enable='gte(t,{max(0.0, dur - 3):.3f})'"
        nxt = f"v{i}"
        parts.append(f"[{cur}][{lbl}]overlay=x={x}:y={y}:format=auto:eof_action=repeat{enable}[{nxt}]")
        cur = nxt
    return ";".join(parts), cur


def render_clip(src: str, clip: dict, ass_path: str, out_path: str, src_info: dict, face_cx: Optional[float],
                job: Job, logos: Optional[list[dict]] = None):
    w, h = src_info["width"], src_info["height"]
    start, end = clip["start"], clip["end"]
    dur = end - start

    target_ratio = 9 / 16
    if w / h > target_ratio:
        crop_w = int(h * target_ratio) // 2 * 2
        cx = (face_cx if face_cx is not None else 0.5) * w
        x = int(min(max(cx - crop_w / 2, 0), w - crop_w))
        crop = f"crop={crop_w}:{h}:{x}:0,"
    else:
        crop = ""
    base = (
        f"[0:v]{crop}scale=1080:1920:force_original_aspect_ratio=decrease,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        f"subtitles='{_ff_escape(ass_path)}':fontsdir='{_ff_escape(str(FONTS_DIR))}'[base]"
    )
    logos = logos or []
    logo_inputs: list[str] = []
    for lg in logos:
        logo_inputs += ["-i", lg["path"]]
    if logos:
        chain, final = logo_filter_chain(logos, first_input_index=1, dur=dur)
        filter_complex = base + ";" + chain
    else:
        filter_complex, final = base, "base"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-hide_banner",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src,
        *logo_inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{final}]", "-map", "0:a:0",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-r", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart", "-threads", str(max(1, os.cpu_count() or 2)),
        out_path,
    ]
    run(cmd, timeout=60 * 30)


def _ff_escape(path: str) -> str:
    # escape for ffmpeg filter option values inside single quotes
    return path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


# --------------------------------------------------------------------------- #
# 5) Output, history
# --------------------------------------------------------------------------- #
def default_hashtags(title: str, text: str) -> str:
    stop = set("the a an and or of to in on for with is are was were be this that it you your we our i at as by from".split())
    toks = re.findall(r"[A-Za-zĀ-ž]{4,}", (title + " " + text))
    freq: dict[str, int] = {}
    for t in toks:
        tl = t.lower()
        if tl in stop:
            continue
        freq[tl] = freq.get(tl, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:4]
    tags = ["#shorts", "#viral"] + [f"#{k}" for k, _ in top]
    return " ".join(dict.fromkeys(tags))


def fill_caption(template: str, clip: dict, title: str) -> str:
    cap = clip.get("caption") or ""
    hashtags = default_hashtags(title, clip.get("text", ""))
    if cap:
        # Gemini wrote the caption: template still wraps it so #Ad is always present
        body = template.format(hook=cap, title=title, hashtags=hashtags if "#" not in cap else "",
                               start=f"{clip['start']:.1f}", end=f"{clip['end']:.1f}")
    else:
        body = template.format(hook=clip.get("hook", ""), title=title, hashtags=hashtags,
                               start=f"{clip['start']:.1f}", end=f"{clip['end']:.1f}")
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    if "#ad" not in body.lower():
        body += "\n#Ad\n"
    return body


def write_captions_md(out_dir: Path, title: str, source: str, clips: list[dict]) -> Path:
    lines = [f"# {title}", "", f"Source: {source}", ""]
    for c in clips:
        lines += [
            f"## {Path(c['file']).name}  ({c['start']:.1f}s – {c['end']:.1f}s, {c['end']-c['start']:.1f}s)",
            "",
            f"**Hook:** {c['hook']}",
            "",
            "**Caption:**",
            "",
            "```",
            c["caption_text"].rstrip(),
            "```",
            "",
            f"<details><summary>Transcript</summary>\n\n{c.get('text','')}\n\n</details>",
            "",
        ]
    p = out_dir / "captions.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def make_zip(out_dir: Path, files: list[Path]) -> Path:
    zp = out_dir / "clips.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    return zp


def load_history() -> list[dict]:
    import gcs

    if gcs.enabled():
        remote = gcs.load_history()
        if remote is not None:
            return remote
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def append_history(entry: dict, keep: int = 20):
    with _history_lock:
        hist = [h for h in load_history() if h.get("id") != entry["id"]]
        hist.insert(0, entry)
        removed = hist[keep:]
        hist = hist[:keep]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
        import gcs

        gcs.save_history(hist)
    # drop output folders that fell off the history
    for old in removed:
        shutil.rmtree(OUT_DIR / old["id"], ignore_errors=True)
        gcs.delete_job(old["id"])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def process(job: Job, url: str = "", local_file: Optional[str] = None, cookies_path: Optional[str] = None,
            mode: str = "local", rules: Optional[dict] = None, logo_overrides: Optional[dict] = None):
    rules = rules or load_rules()
    workdir = WORK_DIR / job.id
    out_dir = job.out_dir
    job.status = "running"
    job.source = url or (Path(local_file).name if local_file else "")
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        job.log(f"Job {job.id} started (mode={mode}, clips={rules['clips_per_video']}, "
                f"{rules['min_len']}-{rules['max_len']}s, whisper={rules.get('whisper_model', DEFAULT_WHISPER_MODEL)})",
                progress=0.01)

        # 1) source
        if local_file:
            src = local_file
            title = Path(local_file).stem
            job.log(f"Using uploaded file: {Path(local_file).name}")
        elif url.strip():
            src, title = download_source(url, workdir, cookies_path, job)
        else:
            raise RuntimeError("Provide a URL or upload a video file")
        job.title = title
        job.log(f"Title: {title}", progress=0.1)
        src = limit_duration(src, workdir, job, float(rules.get("max_video_minutes", DEFAULT_MAX_VIDEO_MINUTES)))
        info = probe_video(src)
        job.log(f"Source: {info['width']}x{info['height']}, {info['duration']/60:.1f} min")

        # 2) transcription
        wav = str(workdir / "audio.wav")
        extract_audio(src, wav)
        job.log("Audio extracted", progress=0.15)
        words, segments, lang = transcribe(wav, job, info["duration"], str(rules.get("whisper_model", DEFAULT_WHISPER_MODEL)))

        # 3) selection
        rms, hop = audio_rms_curve(wav)
        clips: list[dict] = []
        if mode == "gemini":
            try:
                clips = select_moments_gemini(words, segments, rules, info["duration"], title, job)
            except Exception as e:
                job.log(f"Gemini selection failed ({e}); falling back to local scoring.")
        if not clips:
            clips = select_moments_local(words, segments, rms, hop, rules, info["duration"], job)
        job.log(f"{len(clips)} moments selected", progress=0.55)

        # 4) render
        logos = active_logos(rules, logo_overrides, log=job.log)
        if logos:
            job.log("Logos: " + ", ".join(f"{l['slot']} ({l['position']}, {l['show']})" for l in logos))
        rendered: list[Path] = []
        for i, clip in enumerate(clips, start=1):
            name = f"clip_{i:02d}.mp4"
            out_path = out_dir / name
            face_cx = None
            if info["width"] / info["height"] > 9 / 16:
                face_cx = detect_face_center(src, clip["start"], clip["end"], workdir)
                job.log(f"  clip {i}: face {'found at x=%.2f' % face_cx if face_cx is not None else 'not found, centre crop'}")
            ass_path = str(workdir / f"clip_{i:02d}.ass")
            build_ass(clip, rules, ass_path, clip.get("hook", ""))
            job.log(f"  rendering {name} ({clip['start']:.1f}-{clip['end']:.1f}s) ...")
            render_clip(src, clip, ass_path, str(out_path), info, face_cx, job, logos)
            clip["file"] = str(out_path)
            clip["caption_text"] = fill_caption(rules["caption_template"], clip, title)
            rendered.append(out_path)
            job.clips.append({
                "file": str(out_path), "start": clip["start"], "end": clip["end"],
                "hook": clip.get("hook", ""), "caption": clip["caption_text"], "text": clip.get("text", ""),
            })
            job.log(f"  done {name}", progress=0.55 + 0.4 * i / len(clips))

        # 5) outputs
        captions = write_captions_md(out_dir, title, job.source, clips)
        zp = make_zip(out_dir, rendered + [captions])
        job.captions_path = str(captions)
        job.zip_path = str(zp)
        gcs_uris: dict[str, str] = {}
        import gcs

        if gcs.enabled():
            try:
                job.log(f"Uploading clips.zip + captions.md to gs://{gcs.BUCKET}/{gcs.job_prefix(job.id)}/ ...",
                        progress=0.97)
                gcs_uris["zip"] = gcs.upload_file(zp, job.id) or ""
                gcs_uris["captions"] = gcs.upload_file(captions, job.id) or ""
                job.gcs = gcs_uris
            except Exception as e:
                job.log(f"GCS upload failed ({e}) — results are only on local disk.")
        job.status = "done"
        job.finished_at = time.time()
        job.log(f"Finished: {len(rendered)} clips in {out_dir} ({(job.finished_at - job.started_at)/60:.1f} min)",
                progress=1.0)
        append_history({
            "id": job.id, "title": title, "source": job.source, "mode": mode,
            "finished": time.strftime("%Y-%m-%d %H:%M"),
            "clips": [{"file": Path(c["file"]).name, "start": c["start"], "end": c["end"], "hook": c["hook"],
                       "caption": c["caption"]} for c in job.clips],
            "zip": zp.name, "captions": captions.name, "gcs": gcs_uris,
        })
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.finished_at = time.time()
        job.log(f"ERROR: {e}")
        shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        # intermediate files are always removed
        shutil.rmtree(workdir, ignore_errors=True)


def run_in_background(job: Job, **kwargs) -> threading.Thread:
    t = threading.Thread(target=process, args=(job,), kwargs=kwargs, daemon=True)
    t.start()
    return t


if __name__ == "__main__":  # simple CLI for local testing
    import argparse

    ap = argparse.ArgumentParser(description="Clipper CLI")
    ap.add_argument("source", help="URL or local video path")
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--mode", default="local", choices=["local", "gemini"])
    ap.add_argument("--my-logo", default=None, help="PNG path or name in logos/ (overrides rules.yaml)")
    ap.add_argument("--campaign-logo", default=None, help="PNG path or name in logos/ (overrides rules.yaml)")
    args = ap.parse_args()
    overrides = {}
    if args.my_logo is not None:
        overrides["my_logo"] = args.my_logo
    if args.campaign_logo is not None:
        overrides["campaign_logo"] = args.campaign_logo
    job = Job()
    if os.path.exists(args.source):
        process(job, local_file=args.source, cookies_path=args.cookies, mode=args.mode, logo_overrides=overrides)
    else:
        process(job, url=args.source, cookies_path=args.cookies, mode=args.mode, logo_overrides=overrides)
    print(json.dumps({"status": job.status, "error": job.error, "clips": job.clips, "zip": job.zip_path},
                     indent=1, ensure_ascii=False))
    raise SystemExit(0 if job.status == "done" else 1)
