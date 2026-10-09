# ✂️ Clipper

Single-user, CPU-only clipping tool: paste a YouTube / Kick / Google Drive link, a `gs://` path, or upload a
file, and get N vertical 1080×1920 clips with karaoke subtitles, a hook overlay, normalised audio, captions and
a zip. Deployed to **Google Cloud Run** (2 vCPU / 4 GiB, scale-to-zero) from this GitHub repo on every push
to `main`. Designed for a tiny budget: nothing is billed while idle.

| Stage | Tool |
|---|---|
| Download | `yt-dlp` (≤1080p mp4, optional `cookies.txt`), `gdown` for Google Drive, `gs://` from Cloud Storage, or upload |
| Transcription | `faster-whisper` `base` (default) or `small`, int8, CPU, VAD filter, word timestamps |
| Moment selection | **local** heuristic (speech density + RMS peaks + keywords + ?/! − banned words) or **gemini-2.5-flash** (Google AI or Vertex AI express key) |
| Render | `ffmpeg`: 9:16 crop (mediapipe face-centred), ASS karaoke subtitles, hook text (first 3 s), `loudnorm` −14 LUFS, h264 veryfast CRF 23 |
| Output | `out/<id>/clip_01..N.mp4`, `captions.md`, `clips.zip` → `clips.zip` + `captions.md` copied to the GCS bucket |

Sources longer than `max_video_minutes` (default 60) are trimmed. Intermediate files are deleted after every
job; only the last 20 jobs are kept (locally and in the bucket).

---

## 1. Google Cloud setup (one time, ≈15 min)

Use the **same GCP project** that owns the Vertex AI key. Replace `PROJECT_ID`, `BUCKET` below.

### 1.1 Enable APIs

Console: *APIs & Services → Enable APIs* → enable **Cloud Run Admin API**, **Cloud Build API**,
**Artifact Registry API**, **Cloud Storage API**. Or with `gcloud`:

```bash
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com iam.googleapis.com
```

### 1.2 Service account for GitHub Actions (deployer)

Console: *IAM & Admin → Service Accounts → Create* → name `github-deployer`, then grant these roles:

| Role | Why |
|---|---|
| Cloud Run Admin | `gcloud run deploy` |
| Cloud Build Editor | `--source .` builds the image with Cloud Build |
| Artifact Registry Writer | push the built image |
| Service Account User | deploy a service that runs as the runtime SA |
| Storage Object Admin | Cloud Build uploads the source tarball; also lets the deployer manage the clips bucket |

Same with `gcloud`:

```bash
SA=github-deployer@PROJECT_ID.iam.gserviceaccount.com
gcloud iam service-accounts create github-deployer --display-name "GitHub deployer"
for r in roles/run.admin roles/cloudbuild.builds.editor roles/artifactregistry.writer \
         roles/iam.serviceAccountUser roles/storage.objectAdmin; do
  gcloud projects add-iam-policy-binding PROJECT_ID --member "serviceAccount:$SA" --role "$r" --quiet
done
# JSON key → GitHub secret GCP_SA_KEY
gcloud iam service-accounts keys create gcp-sa-key.json --iam-account "$SA"
```

### 1.3 Runtime permissions (the Cloud Run service itself)

By default Cloud Run runs as the **Compute Engine default service account**
(`PROJECT_NUMBER-compute@developer.gserviceaccount.com`). It needs:

| Role | Why |
|---|---|
| Storage Object Admin (on the bucket) | upload `clips.zip` / `captions.md` / `history.json`, delete old jobs |
| Service Account Token Creator (on itself) | sign download URLs (IAM `signBlob`) — without it links fall back to local downloads |

```bash
PN=$(gcloud projects describe PROJECT_ID --format 'value(projectNumber)')
RT=$PN-compute@developer.gserviceaccount.com
gcloud storage buckets add-iam-policy-binding gs://BUCKET --member "serviceAccount:$RT" --role roles/storage.objectAdmin
gcloud iam service-accounts add-iam-policy-binding "$RT" --member "serviceAccount:$RT" --role roles/iam.serviceAccountTokenCreator
```

(Optionally create a dedicated runtime SA with just these roles and set the GitHub variable `CLOUD_RUN_SA` to its e-mail;
the deployer then also needs *Service Account User* on it.)

### 1.4 Bucket for results

```bash
gcloud storage buckets create gs://BUCKET --location europe-west1 --uniform-bucket-level-access
# auto-delete objects after 30 days (keeps storage cost ≈ 0)
cat > lifecycle.json <<'EOF'
{"rule":[{"action":{"type":"Delete"},"condition":{"age":30}}]}
EOF
gcloud storage buckets update gs://BUCKET --lifecycle-file=lifecycle.json
```

Keep the bucket **private**. Downloads use short-lived signed URLs (2 h by default, `GCS_SIGNED_URL_MINUTES`).

### 1.5 Budget alert

Console: *Billing → Budgets & alerts → Create budget* → scope: this project → amount **€5** →
thresholds 50 % / 90 % / 100 % → e-mail alerts. (A budget alert only notifies; to hard-stop spending you would
also need a Cloud Function that disables billing — overkill at this scale.)

### 1.6 Keep the image registry small (optional)

Cloud Run `--source` stores images in Artifact Registry repo `cloud-run-source-deploy`. The image is ≈3.5 GB,
so add a cleanup policy that keeps only the 2 newest versions:

```bash
cat > policy.json <<'EOF'
[{"name":"keep-2","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":2}},
 {"name":"delete-rest","action":{"type":"Delete"},"condition":{"olderThan":"1d"}}]
EOF
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
    --location europe-west1 --policy policy.json --no-dry-run
```

---

## 2a. Manual deploy from your machine: `./deploy.sh`

```bash
gcloud auth login && gcloud config set project PROJECT_ID
./deploy.sh 'YOUR_PASSWORD' 'GEMINI_API_KEY' 'BUCKET'      # 2nd and 3rd argument optional
```

The script enables the Cloud Run / Cloud Build / Artifact Registry / Cloud Storage APIs and runs the same
`gcloud run deploy clipper --source . --region europe-west1 --cpu 2 --memory 4Gi --timeout 3600
--min-instances 0 --max-instances 1 --concurrency 4 --allow-unauthenticated` as the GitHub Action,
with `PASSWORD`, `GEMINI_API_KEY` and `GCS_BUCKET` set as env vars. Override the region/service with
`GCP_REGION` / `SERVICE`, the runtime service account with `CLOUD_RUN_SA`.

## 2b. GitHub → Cloud Run (auto-deploy on every push to `main`)

The workflow `.github/workflows/deploy-cloudrun.yml` authenticates with `google-github-actions/auth`
and runs

```
gcloud run deploy clipper --source . --region europe-west1 --cpu 2 --memory 4Gi --timeout 3600 \
  --min-instances 0 --max-instances 1 --concurrency 4 --allow-unauthenticated --env-vars-file env.yaml
```

(CPU is throttled outside requests — that is the cheap default; the app is built so all work happens inside
the request, see §4.)

Configure the repo: **Settings → Secrets and variables → Actions**

| Kind | Name | Value |
|---|---|---|
| Secret | `GCP_SA_KEY` | contents of `gcp-sa-key.json` (§1.2) |
| Secret | `PASSWORD` | UI login password (username `admin`) |
| Secret | `GEMINI_API_KEY` | optional — Google AI Studio key **or** Vertex AI express key (`AQ.…`, auto-detected) |
| Variable | `GCP_PROJECT_ID` | your project id |
| Variable | `GCS_BUCKET` | `BUCKET` from §1.4 (leave empty → local disk only) |
| Variable | `GCP_REGION` | optional, default `europe-west1` |
| Variable | `CLOUD_RUN_SA` | optional runtime service-account e-mail |

Then push to `main` (or *Actions → Deploy to Cloud Run → Run workflow*). The first build takes ≈10–15 min
(pip install + baking the whisper models into the image); later builds reuse layers. The service URL is
printed at the end of the job and in *Cloud Run → clipper*.

The workflow is skipped while `GCP_PROJECT_ID` is unset, so the repo also works without GCP.

**Rotate the Vertex key** if it was ever shared in chat/e-mail: *Google Cloud → APIs & Services → Credentials*.

---

## 3. Using the app

- **Process**: paste a link (YouTube / Kick / Google Drive / `gs://bucket/file.mp4`) *or* upload a file,
  optionally add `cookies.txt`, pick `local`/`gemini`, press **Process**. The log streams live. When done you
  get previews, per-clip downloads, **Download all (zip)**, `captions.md` and (with a bucket) signed Cloud
  Storage links.
  **Keep the tab open** — the job runs inside that request (closing the tab lets Cloud Run throttle the CPU
  and the job crawls).
- **Rules**: edit `rules.yaml` and **Save** (validated). Applies to the next job. Edits live on the instance's
  disk; commit the file to make them permanent across deploys.
- **History**: last 20 jobs; ☁️ entries have signed links to `clips.zip`/`captions.md` in the bucket, 💾
  entries exist only while the instance lives.

### Logos

`rules.yaml` → `logos:` has two slots, `my_logo` (default corner bottom-right) and `campaign_logo` (top-left).
Each has `enabled`, `file` (name in `logos/` or an absolute path), `position` (`top_left|top_right|bottom_left|bottom_right`),
`width_pct` (12 % of the 1080 px frame), `opacity` (0.85), `margin_px` (40) and `show` (`always|first_3s|last_3s`).
Logos are overlaid **after** subtitles and hook with an ffmpeg `filter_complex` (`scale` → `overlay`), and are
kept out of the text bands: top-corner logos are capped so they end above y = 200 (the hook band starts at
y ≈ 230), bottom-corner logos start below y = 1500 (subtitles end at y ≈ 1440). Two enabled logos must use
different corners.

In the **Process** tab the *Logos* accordion has, per slot, an upload field and a dropdown of saved logos.
An uploaded PNG is stored in `logos/` (and in `gs://BUCKET/clipper/logos/` when a bucket is configured) so
it can be picked from the list next time without re-uploading. Upload or a picked name enables the logo for
that job; `(none)` disables it; `(rules.yaml default)` follows the rules file. CLI: `--my-logo`, `--campaign-logo`.

### Uploads on Cloud Run

Cloud Run rejects HTTP/1 request bodies above 32 MiB, so the upload field is limited to 30 MB there.
For bigger sources use a link, or put the file in the bucket and paste its `gs://` path:

```bash
gcloud storage cp my_video.mp4 gs://BUCKET/uploads/
# then paste  gs://BUCKET/uploads/my_video.mp4
```

### YouTube "Sign in to confirm you're not a bot"

Cloud IPs are often blocked by YouTube. Export cookies with a browser extension such as
*Get cookies.txt LOCALLY* (Netscape format) while logged in with a throw-away account and upload the file.
Kick, Google Drive and `gs://` sources need no cookies.

---

## 4. Cost model (why this stays within €10)

Cloud Run is billed **only while a request is being served** (`--min-instances 0`, CPU throttled when idle),
so idle cost is €0. Prices below are Cloud Run tier-1 (`europe-west1`) list prices; the free tier
(180 000 vCPU-s + 360 000 GiB-s per month) is not even subtracted.

| Item | Rate | Per job (20-min source, `base`, 4 clips ≈ 8 min of request) | Per job (60-min source, `base` ≈ 25 min) |
|---|---|---|---|
| vCPU 2 × $0.000024/s | | $0.023 | $0.072 |
| Memory 4 GiB × $0.0000025/s | | $0.005 | $0.015 |
| Gemini 2.5 Flash (Vertex) | $0.30 / M input tokens | < $0.01 | < $0.01 |
| Cloud Storage (zip ≈ 60–150 MB, 30-day lifecycle) | $0.02 / GB·month | < $0.01 | < $0.01 |
| Download egress | $0.12 / GB | ≈ $0.01 | ≈ $0.02 |
| **≈ total** | | **≈ €0.04** | **≈ €0.10** |

Fixed: Artifact Registry ≈ €0.30/month for the image (with the keep-2 policy), Cloud Build free tier covers
the builds (≈10 min each). Realistic monthly total for 20–40 videos: **€1–3**. Using `whisper_model: small`
roughly triples transcription time (and cost); `max_video_minutes` caps the worst case.

Why the job runs inside the request: with the default (cheaper) CPU throttling, Cloud Run gives the container
full CPU only while an HTTP request is open. The **Process** click is a Gradio generator that streams the log
over SSE for the whole job, so the CPU stays allocated without paying for `--no-cpu-throttling`
(which would bill 2 vCPU 24/7 ≈ €120/month).

---

## 5. Test locally

### Docker (same image as Cloud Run)

```bash
docker build -t clipper .

docker run --rm -p 8080:8080 \
  -e PASSWORD=test123 \
  -e GEMINI_API_KEY=...            # optional
  -e GCS_BUCKET=BUCKET -e GOOGLE_APPLICATION_CREDENTIALS=/key.json -v "$PWD/gcp-sa-key.json:/key.json:ro" \  # optional
  -v "$PWD/out:/home/user/app/out" \
  clipper
```

Open <http://localhost:8080>, log in with `admin` / `test123`, upload a short video with speech and press **Process**.

### Without Docker (venv)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg must be on PATH; deno or node is needed for YouTube downloads
PASSWORD=test123 python app.py                 # UI on http://localhost:7860
python pipeline.py path/to/video.mp4           # CLI, prints JSON with the produced clips
python pipeline.py "https://www.youtube.com/watch?v=..." --cookies cookies.txt --mode gemini
```

### What to expect (measured on a 2.9-min 720p test video, 4 CPU threads)

```
Job ... started (mode=local, clips=4, 15-30s, whisper=base)
Transcription done: 393 words, language=en
Selected windows: 9.7-38.0s, 43.9-67.0s, 95.4-123.8s, 138.5-161.4s
Finished: 4 clips in out/<id> (≈1 min)

$ ls out/<id>
captions.md  clip_01.mp4  clip_02.mp4  clip_03.mp4  clip_04.mp4  clips.zip
```

Each clip: `h264 1080x1920 30 fps`, integrated loudness ≈ −14.5 LUFS. In `gemini` mode (Vertex express
key) Gemini picked 4 windows and wrote hooks such as *"Free video clipper? No GPU!"* plus hashtagged captions.

---

## Files

```
app.py            Gradio UI (Process / Rules / History); job streams inside the request
pipeline.py       download → transcribe → select → render → zip (+ GCS upload); also a CLI
gcs.py            optional Cloud Storage persistence + signed URLs
rules.yaml        editable rules (whisper_model, max_video_minutes, lengths, subtitles, caption template …)
requirements.txt  Python deps
Dockerfile        python:3.11-slim + ffmpeg + fonts + deno + whisper base/small baked in; listens on $PORT
prefetch_models.py  build-time whisper model download with retries (non-fatal)
deploy.sh         manual one-shot deploy (enables APIs + gcloud run deploy)
.github/workflows/deploy-cloudrun.yml   gcloud run deploy on push to main
fonts/, models/   Montserrat-Bold.ttf (SIL OFL), mediapipe BlazeFace model (Apache-2.0)
logos/            saved logo PNGs (uploads land here; mirrored to the bucket)
```

## Build troubleshooting (Cloud Build)

The image is built by Cloud Build's classic Docker builder, so the Dockerfile avoids BuildKit-only syntax.
Things that were made robust after a failed build:

- **PyAV**: `faster-whisper` imports `av`; `av` 14.3+/15+ publish no CPython 3.11 Linux wheels, so an
  unpinned `av` makes pip compile it from source (needs ffmpeg headers) and the build dies right after the
  apt step. `requirements.txt` pins `av<=14.2.0` on Python < 3.12 and the Dockerfile installs with
  `--only-binary=:all:` so any missing wheel fails loudly instead of compiling.
- **deno** (JS runtime for yt-dlp's YouTube extractor): installed from the official PyPI wheel (`pip install deno`),
  with the install script as fallback; both are optional and never fail the build.
- **Whisper models**: `prefetch_models.py` downloads `base` and `small` with retries and never fails the
  build (missing models are fetched on first use). The download runs as UID 1000 so no large `chown` layer.
- **Fonts / face model**: committed in the repo; network fallbacks are non-fatal.
- Lint: `hadolint Dockerfile` is clean. Dependency set verified with a real CPython 3.11 venv
  (`pip install --only-binary=:all: -r requirements.txt` → imports OK, pipeline run OK).

To read a failed build log from a laptop: *Cloud Build → History → the red build → Build log*, or
`gcloud builds list --region europe-west1 --limit 3` and `gcloud builds log BUILD_ID --region europe-west1`.

## Notes & limits

- One job at a time (single user, `--max-instances 1`). A second **Process** click while a job runs is refused.
- Cloud Run request timeout is 60 min; `max_video_minutes: 60` with `base` keeps jobs well under that. If a job
  could exceed it, lower `max_video_minutes` or `clips_per_video`.
- Local disk is ephemeral: anything not in the bucket disappears when the instance is replaced.
- Cold start after idle ≈ 20–40 s (large image + model load).
