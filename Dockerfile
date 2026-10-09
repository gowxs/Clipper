# Clipper — Google Cloud Run (2 vCPU / 4 GiB, CPU only). Also works with plain `docker run`.
#
# Built by Cloud Build's classic docker builder (no BuildKit): no heredocs, no --mount, plain RUN lines.
# Every network step that is not strictly required (deno, model prefetch, font fallback) is non-fatal.
FROM python:3.11-slim

# bash with pipefail so a failing command inside a pipe fails the step (hadolint DL4006)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# ---- system packages: ffmpeg, fontconfig + fallback fonts, OpenGL runtime for mediapipe, curl
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        fontconfig \
        fonts-dejavu-core \
        fonts-liberation \
        libgl1 \
        libegl1 \
        libgles2 \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- non-root user (Cloud Run does not require it, but nothing here needs root at runtime)
RUN useradd --create-home --uid 1000 --shell /bin/bash user

ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=8080 \
    OMP_NUM_THREADS=2

WORKDIR /home/user/app

# ---- python deps (all pinned to versions that ship manylinux wheels for CPython 3.11: no compiling)
COPY requirements.txt ./
# hadolint ignore=DL3013
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --only-binary=:all: -r requirements.txt \
    && python -c "import gradio, faster_whisper, mediapipe, yt_dlp, av, google.genai, google.cloud.storage; print('deps ok')"

# ---- deno: JS runtime for yt-dlp's YouTube extractor. Optional: yt-dlp still works without it
#      (some YouTube formats may be missing). Try the official PyPI wheel, then the install script.
# hadolint ignore=DL3013
RUN (python -m pip install "deno>=2,<3" && deno --version) \
    || (curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh && deno --version) \
    || echo "WARNING: deno not installed; continuing without a JS runtime"

# ---- subtitle font + mediapipe face model (both committed in the repo; downloads are only a fallback)
COPY --chown=1000:1000 fonts/ ./fonts/
COPY --chown=1000:1000 models/ ./models/
RUN mkdir -p /usr/local/share/fonts/clipper \
    && if [ ! -s fonts/Montserrat-Bold.ttf ]; then \
         url="$(curl -fsSL -A 'Mozilla/5.0' 'https://fonts.googleapis.com/css2?family=Montserrat:wght@700' | grep -o 'https://[^)]*\.ttf' | head -1)" \
         && curl -fsSL -o fonts/Montserrat-Bold.ttf "$url" || echo "WARNING: Montserrat download failed; libass will fall back to DejaVu/Liberation"; \
       fi \
    && if [ ! -s models/blaze_face_short_range.tflite ]; then \
         curl -fsSL -o models/blaze_face_short_range.tflite \
           "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" \
         || echo "WARNING: face model download failed; clips will use centre crop"; \
       fi \
    && (cp fonts/*.ttf /usr/local/share/fonts/clipper/ 2>/dev/null || true) \
    && fc-cache -f \
    && (fc-list | grep -i montserrat || echo "WARNING: Montserrat not registered")

# ---- switch to the runtime user BEFORE the big model download so the files are owned by it
#      (a later chown -R would duplicate ~700 MB into another layer)
RUN mkdir -p /home/user/app/out /home/user/app/work /home/user/app/logos "$HF_HOME" \
    && chown -R 1000:1000 /home/user
USER 1000:1000

# ---- pre-download whisper models (base + small) with retries; never fails the build
COPY --chown=1000:1000 prefetch_models.py ./
RUN python prefetch_models.py base small

# ---- application code
COPY --chown=1000:1000 . ./

# Cloud Run injects $PORT (default 8080); app.py listens on it
EXPOSE 8080
CMD ["python", "app.py"]
