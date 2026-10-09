"""
Pre-download faster-whisper models into the image at build time, with retries.

Never fails the build: if the Hugging Face Hub is unreachable or rate-limits the download, the models
are fetched lazily on the first job instead (slower first run, same result).
Usage: python prefetch_models.py [model ...]   (default: base small)
"""
import sys
import time

models = sys.argv[1:] or ["base", "small"]
ok = True
for name in models:
    for attempt in range(1, 5):
        try:
            from faster_whisper import download_model

            path = download_model(name)
            print(f"[prefetch] {name}: ok -> {path}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            wait = 10 * attempt
            print(f"[prefetch] {name}: attempt {attempt} failed ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        ok = False
        print(f"[prefetch] {name}: giving up — it will be downloaded on first use", flush=True)
print("[prefetch] done" + ("" if ok else " (with missing models)"), flush=True)
sys.exit(0)
