# Clipper — Hugging Face Docker Space (free CPU tier)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: ffmpeg, fonts, OpenGL runtime for mediapipe, curl/unzip for the font + deno
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fontconfig \
        fonts-dejavu-core \
        fonts-liberation \
        libgl1 \
        libegl1 \
        libgles2 \
        libglib2.0-0 \
        ca-certificates \
        curl \
        unzip \
        git \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JavaScript runtime for YouTube (deno is the recommended one)
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh \
    && deno --version

# HF Spaces run as UID 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Subtitle font + mediapipe face model are committed in the repo (fonts/, models/);
# download them only if missing (e.g. when building from a partial checkout).
COPY --chown=user fonts/ fonts/
COPY --chown=user models/ models/
RUN set -e; \
    if [ ! -f fonts/Montserrat-Bold.ttf ]; then \
        url=$(curl -fsSL -A "Mozilla/5.0" "https://fonts.googleapis.com/css2?family=Montserrat:wght@700" | grep -o 'https://[^)]*\.ttf' | head -1); \
        curl -fsSL -o fonts/Montserrat-Bold.ttf "$url"; \
    fi; \
    if [ ! -f models/blaze_face_short_range.tflite ]; then \
        curl -fsSL -o models/blaze_face_short_range.tflite \
          "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"; \
    fi; \
    mkdir -p /usr/local/share/fonts && cp fonts/*.ttf /usr/local/share/fonts/ && fc-cache -f \
    && fc-list | grep -i montserrat

# Pre-download the whisper model so the first job does not wait for it
RUN python -c "from faster_whisper import download_model; download_model('small')" \
    && chown -R user:user /home/user/.cache

COPY --chown=user . .
RUN mkdir -p out work && chown -R user:user /home/user/app

USER user
EXPOSE 7860
CMD ["python", "app.py"]
