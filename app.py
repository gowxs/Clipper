"""
Clipper — single-user Gradio UI for turning long videos into vertical clips.

Runs on Google Cloud Run (CPU only, 2 vCPU / 4 GiB, request-based billing).  The whole job runs *inside*
the HTTP request of the "Process" click (a Gradio generator that streams the log), so Cloud Run keeps the
CPU allocated for the duration; nothing heavy runs in detached background threads.
"""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import gradio as gr

import gcs
import pipeline as pl

MAX_SLOTS = 8                      # max clips shown in the UI (clips_per_video is capped to this)
PASSWORD = os.environ.get("PASSWORD", "").strip()
USERNAME = os.environ.get("USERNAME", "admin").strip() or "admin"
HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY", "").strip())
ON_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))
# Cloud Run rejects HTTP/1 request bodies above 32 MiB → larger sources go via link / gs:// / Drive
MAX_UPLOAD = os.environ.get("MAX_UPLOAD", "30mb" if ON_CLOUD_RUN else "4gb")

pl.OUT_DIR.mkdir(parents=True, exist_ok=True)
pl.WORK_DIR.mkdir(parents=True, exist_ok=True)

_state_lock = threading.Lock()
_current_job: pl.Job | None = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _file_path(f) -> str | None:
    """Gradio File values can be str, dict or objects with .name."""
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get("path") or f.get("name")
    return getattr(f, "path", None) or getattr(f, "name", None)


def _slot_updates(clips: list[dict]):
    out = []
    for i in range(MAX_SLOTS):
        if i < len(clips):
            c = clips[i]
            name = Path(c["file"]).name
            md = (f"**{name}** · {c['start']:.1f}s – {c['end']:.1f}s ({c['end'] - c['start']:.1f}s)\n\n"
                  f"**Hook:** {c.get('hook', '')}\n\n```\n{c.get('caption', '').rstrip()}\n```")
            out += [gr.Video(value=c["file"], visible=True, label=name),
                    gr.Markdown(value=md, visible=True),
                    gr.DownloadButton(value=c["file"], visible=True, label=f"Download {name}")]
        else:
            out += [gr.Video(value=None, visible=False), gr.Markdown(value="", visible=False),
                    gr.DownloadButton(value=None, visible=False)]
    return out


def _links_md(job_id: str, title: str = "") -> str:
    """Markdown with signed GCS URLs for a job (empty string when GCS is off / signing fails)."""
    if not gcs.enabled():
        return ""
    parts = []
    zip_url = gcs.signed_url(gcs.blob_name(job_id, "clips.zip"), download_name=f"{pl.sanitize_filename(title or job_id)}_clips.zip")
    cap_url = gcs.signed_url(gcs.blob_name(job_id, "captions.md"), download_name="captions.md")
    if zip_url:
        parts.append(f"[⬇️ clips.zip (Cloud Storage, link valid {gcs.SIGNED_URL_MINUTES} min)]({zip_url})")
    if cap_url:
        parts.append(f"[⬇️ captions.md]({cap_url})")
    if not parts:
        return f"☁️ Stored in `gs://{gcs.BUCKET}/{gcs.job_prefix(job_id)}/` (signed URLs unavailable — see README)"
    return " · ".join(parts)


def _frame(job: pl.Job | None, status: str, final: bool = False):
    """One UI update tuple: log, status, zip btn, captions btn, links md, slots..."""
    log = job.log_text if job else ""
    if not final or job is None or job.status != "done":
        return (log, status, gr.DownloadButton(visible=False), gr.DownloadButton(visible=False),
                gr.Markdown(value="", visible=False), *_slot_updates([]))
    links = _links_md(job.id, job.title)
    return (log, status,
            gr.DownloadButton(value=job.zip_path, visible=bool(job.zip_path), label="⬇️ Download all (zip)"),
            gr.DownloadButton(value=job.captions_path, visible=bool(job.captions_path), label="⬇️ captions.md"),
            gr.Markdown(value=links, visible=bool(links)),
            *_slot_updates(job.clips))


# --------------------------------------------------------------------------- #
# Process tab — generator keeps the HTTP request (and Cloud Run CPU) alive until the job ends
# --------------------------------------------------------------------------- #
def run_processing(url: str, video_file, cookies_file, mode: str):
    global _current_job
    url = (url or "").strip()
    video_path = _file_path(video_file)
    cookies_path = _file_path(cookies_file)

    with _state_lock:
        if _current_job is not None and _current_job.status in ("queued", "running"):
            yield _frame(_current_job, "⏳ A job is already running — wait for it to finish.")
            return
        if not url and not video_path:
            yield _frame(None, "⚠️ Paste a link or upload a video first.")
            return

        rules = pl.load_rules()
        rules["clips_per_video"] = min(int(rules["clips_per_video"]), MAX_SLOTS)
        if mode == "gemini" and not HAS_GEMINI:
            mode = "local"

        job = pl.Job()
        job_work = pl.WORK_DIR / job.id
        if video_path:                       # move uploads out of Gradio's cache so cleanup is ours
            job_work.mkdir(parents=True, exist_ok=True)
            orig = Path(video_path)
            dst = job_work / (pl.sanitize_filename(orig.stem) + orig.suffix.lower())
            shutil.copy(video_path, dst)
            video_path = str(dst)
        if cookies_path:
            job_work.mkdir(parents=True, exist_ok=True)
            dst = job_work / "cookies.txt"
            shutil.copy(cookies_path, dst)
            cookies_path = str(dst)
        _current_job = job

    # The worker thread does the work; this generator streams its log every second and only returns
    # when the job is finished, so the request stays open for Cloud Run.
    worker = pl.run_in_background(job, url=url, local_file=video_path, cookies_path=cookies_path,
                                  mode=mode, rules=rules)
    yield _frame(job, f"🚀 Started job {job.id}")
    last_len = -1
    while worker.is_alive():
        worker.join(timeout=1.0)
        n = len(job.log_lines)
        if n != last_len:
            last_len = n
            yield _frame(job, f"⏳ Running… {int(job.progress * 100)}%")
    if job.status == "done":
        yield _frame(job, f"✅ Done — {len(job.clips)} clips", final=True)
    else:
        yield _frame(job, f"❌ Failed: {job.error}", final=True)


def attach_running():
    """On page (re)load: if a job is still running on this instance, show its log."""
    job = _current_job
    if job is None:
        return _frame(None, "Idle")
    if job.status in ("queued", "running"):
        return _frame(job, f"⏳ A job is running ({int(job.progress * 100)}%) — results appear in the tab that started it.")
    if job.status == "done":
        return _frame(job, f"✅ Last job: {len(job.clips)} clips", final=True)
    return _frame(job, f"❌ Last job failed: {job.error}")


# --------------------------------------------------------------------------- #
# Rules tab
# --------------------------------------------------------------------------- #
def save_rules(text: str):
    try:
        merged = pl.save_rules_text(text)
        return (f"✅ Saved. Effective: {merged['clips_per_video']} clips × {merged['min_len']}–{merged['max_len']}s, "
                f"whisper `{merged['whisper_model']}`, max {merged['max_video_minutes']} min")
    except Exception as e:
        return f"❌ Not saved — invalid rules: {e}"


def reload_rules():
    return pl.read_rules_text(), "Reloaded from disk."


def reset_rules():
    text = pl.DEFAULT_RULES_TEXT
    pl.save_rules_text(text)
    return text, "Defaults restored and saved."


# --------------------------------------------------------------------------- #
# History tab
# --------------------------------------------------------------------------- #
def history_table():
    rows = []
    for h in pl.load_history():
        rows.append([h.get("finished", ""), h.get("id", ""), h.get("title", "")[:60], h.get("mode", ""),
                     len(h.get("clips", [])), "☁️" if h.get("gcs", {}).get("zip") else "💾",
                     h.get("source", "")[:80]])
    return rows


def history_choices():
    return [f"{h.get('finished','')} · {h.get('title','')[:50]} · {h.get('id','')}" for h in pl.load_history()]


def refresh_history():
    choices = history_choices()
    return history_table(), gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def show_history_entry(choice: str | None):
    if not choice:
        return gr.File(value=None), gr.DownloadButton(visible=False), ""
    job_id = choice.rsplit("·", 1)[-1].strip()
    entry = next((h for h in pl.load_history() if h.get("id") == job_id), None)
    if not entry:
        return gr.File(value=None), gr.DownloadButton(visible=False), "Entry not found."
    out_dir = pl.OUT_DIR / job_id
    local_clips = [str(out_dir / c["file"]) for c in entry.get("clips", []) if (out_dir / c["file"]).exists()]
    zp = out_dir / entry.get("zip", "clips.zip")

    md = f"### {entry.get('title','')}\n\nSource: {entry.get('source','')} · {entry.get('finished','')} · mode `{entry.get('mode','')}`\n\n"
    links = _links_md(job_id, entry.get("title", "")) if entry.get("gcs", {}).get("zip") else ""
    if links:
        md += links + "\n\n"
    elif not zp.exists():
        md += "_Files of this job are no longer on this instance (ephemeral disk) and were not uploaded to Cloud Storage._\n\n"
    for c in entry.get("clips", []):
        md += f"- **{c['file']}** {c['start']:.1f}–{c['end']:.1f}s — {c.get('hook','')}\n"

    captions_text = None
    cap = out_dir / entry.get("captions", "captions.md")
    if cap.exists():
        captions_text = cap.read_text(encoding="utf-8")
    elif entry.get("gcs", {}).get("captions"):
        captions_text = gcs.read_text(gcs.blob_name(job_id, "captions.md"))
    if captions_text:
        md += "\n<details><summary>captions.md</summary>\n\n```\n" + captions_text + "\n```\n</details>"

    return (gr.File(value=local_clips if local_clips else None),
            gr.DownloadButton(value=str(zp) if zp.exists() else None, visible=zp.exists()), md)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
storage_note = (f"Results are copied to `gs://{gcs.BUCKET}` after every job." if gcs.enabled()
                else "GCS_BUCKET is not set — results live only on this instance's disk until it restarts.")
upload_note = (f"Upload limit {MAX_UPLOAD} (Cloud Run request limit). For bigger files paste a YouTube / Kick / "
               f"Google Drive link or a `gs://bucket/file.mp4` path." if ON_CLOUD_RUN else f"Upload limit {MAX_UPLOAD}.")

with gr.Blocks(title="Clipper") as demo:
    gr.Markdown(f"# ✂️ Clipper\nLong video → vertical clips with karaoke subtitles. CPU only. {storage_note}")

    with gr.Tabs():
        # ---------------- Process ----------------
        with gr.Tab("Process"):
            with gr.Row():
                with gr.Column(scale=3):
                    url_in = gr.Textbox(label="YouTube / Kick / Google Drive link or gs:// path",
                                        placeholder="https://www.youtube.com/watch?v=...   https://kick.com/...   "
                                                    "https://drive.google.com/file/d/.../view   gs://bucket/video.mp4")
                    video_in = gr.File(label=f"… or upload a video file ({upload_note})",
                                       file_types=["video", ".mp4", ".mkv", ".mov", ".webm"], file_count="single")
                with gr.Column(scale=2):
                    cookies_in = gr.File(label="cookies.txt (optional, Netscape format — bypasses YouTube bot checks)",
                                         file_types=[".txt"], file_count="single")
                    mode_in = gr.Radio(
                        choices=["local", "gemini"] if HAS_GEMINI else ["local"],
                        value="gemini" if HAS_GEMINI else "local",
                        label="Moment selection",
                        info=("local = free heuristic scoring; gemini = gemini-2.5-flash picks moments + writes hooks/captions"
                              if HAS_GEMINI else "Set the GEMINI_API_KEY secret to enable Gemini selection"),
                    )
                    process_btn = gr.Button("▶️ Process", variant="primary", size="lg")
                    gr.Markdown("Keep this tab open while processing — the job runs inside this request.")
            status_md = gr.Markdown("Idle")
            log_box = gr.Textbox(label="Progress log", lines=12, max_lines=20, interactive=False, autoscroll=True)
            with gr.Row():
                zip_btn = gr.DownloadButton("⬇️ Download all (zip)", visible=False)
                captions_btn = gr.DownloadButton("⬇️ captions.md", visible=False)
            links_md = gr.Markdown("", visible=False)

            gr.Markdown("### Clips")
            slot_components = []
            for _row in range((MAX_SLOTS + 3) // 4):
                with gr.Row():
                    for _col in range(4):
                        with gr.Column():
                            v = gr.Video(visible=False, height=420, interactive=False)
                            m = gr.Markdown(visible=False)
                            d = gr.DownloadButton(visible=False, size="sm")
                            slot_components += [v, m, d]

            outputs = [log_box, status_md, zip_btn, captions_btn, links_md, *slot_components]
            process_btn.click(run_processing, inputs=[url_in, video_in, cookies_in, mode_in], outputs=outputs,
                              concurrency_limit=1, concurrency_id="process")

        # ---------------- Rules ----------------
        with gr.Tab("Rules"):
            gr.Markdown("Edit `rules.yaml`. Changes apply to the next job (saved on this instance's disk; "
                        "commit the file to the repo to make them permanent across deploys).")
            rules_box = gr.Code(value=pl.read_rules_text(), language="yaml", lines=30, label="rules.yaml")
            with gr.Row():
                save_btn = gr.Button("💾 Save", variant="primary")
                reload_btn = gr.Button("↩️ Reload from disk")
                reset_btn = gr.Button("♻️ Restore defaults")
            rules_status = gr.Markdown("")
            save_btn.click(save_rules, inputs=[rules_box], outputs=[rules_status])
            reload_btn.click(reload_rules, outputs=[rules_box, rules_status])
            reset_btn.click(reset_rules, outputs=[rules_box, rules_status])

        # ---------------- History ----------------
        with gr.Tab("History"):
            gr.Markdown("Last 20 processed videos (☁️ = stored in Cloud Storage, 💾 = local disk only). "
                        "Older jobs are deleted automatically.")
            hist_refresh = gr.Button("🔄 Refresh")
            hist_table = gr.Dataframe(headers=["finished", "id", "title", "mode", "clips", "storage", "source"],
                                      value=[], interactive=False, wrap=True)
            hist_pick = gr.Dropdown(choices=[], value=None, label="Open entry")
            hist_md = gr.Markdown("")
            hist_files = gr.File(label="Clips (only while still on this instance)", file_count="multiple", interactive=False)
            hist_zip = gr.DownloadButton("⬇️ Download all (zip, local)", visible=False)
            hist_refresh.click(refresh_history, outputs=[hist_table, hist_pick])
            hist_pick.change(show_history_entry, inputs=[hist_pick], outputs=[hist_files, hist_zip, hist_md])

    demo.load(refresh_history, outputs=[hist_table, hist_pick])
    demo.load(attach_running, outputs=outputs)


if __name__ == "__main__":
    auth = (USERNAME, PASSWORD) if PASSWORD else None
    if auth is None:
        print("WARNING: PASSWORD env var not set — UI is unprotected!", flush=True)
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        auth=auth,
        auth_message="Clipper — enter the password" if auth else None,
        allowed_paths=[str(pl.OUT_DIR)],
        max_file_size=MAX_UPLOAD,
        show_error=True,
    )
