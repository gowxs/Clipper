"""
Clipper — single-user Gradio UI for turning long videos into vertical clips.
Runs on a Hugging Face Docker Space (CPU only).
"""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import gradio as gr

import pipeline as pl

MAX_SLOTS = 8                      # max clips shown in the UI (clips_per_video is capped to this)
PASSWORD = os.environ.get("PASSWORD", "").strip()
USERNAME = os.environ.get("USERNAME", "admin").strip() or "admin"
HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY", "").strip())

pl.OUT_DIR.mkdir(parents=True, exist_ok=True)
pl.WORK_DIR.mkdir(parents=True, exist_ok=True)

_state_lock = threading.Lock()
_current_job: pl.Job | None = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _file_path(f) -> str | None:
    """Gradio File/Video values can be str, dict or objects with .name."""
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get("path") or f.get("name")
    return getattr(f, "path", None) or getattr(f, "name", None)


def _empty_slots():
    """Hidden result slots."""
    out = []
    for _ in range(MAX_SLOTS):
        out += [gr.Video(value=None, visible=False), gr.Markdown(value="", visible=False),
                gr.DownloadButton(value=None, visible=False)]
    return out


def _filled_slots(clips: list[dict]):
    out = []
    for i in range(MAX_SLOTS):
        if i < len(clips):
            c = clips[i]
            md = (f"**{Path(c['file']).name}** · {c['start']:.1f}s – {c['end']:.1f}s "
                  f"({c['end'] - c['start']:.1f}s)\n\n**Hook:** {c.get('hook', '')}\n\n"
                  f"```\n{c.get('caption', '').rstrip()}\n```")
            out += [gr.Video(value=c["file"], visible=True, label=Path(c["file"]).name),
                    gr.Markdown(value=md, visible=True),
                    gr.DownloadButton(value=c["file"], visible=True, label=f"Download {Path(c['file']).name}")]
        else:
            out += [gr.Video(value=None, visible=False), gr.Markdown(value="", visible=False),
                    gr.DownloadButton(value=None, visible=False)]
    return out


# --------------------------------------------------------------------------- #
# Process tab
# --------------------------------------------------------------------------- #
def start_processing(url: str, video_file, cookies_file, mode: str):
    global _current_job
    url = (url or "").strip()
    video_path = _file_path(video_file)
    cookies_path = _file_path(cookies_file)

    with _state_lock:
        if _current_job is not None and _current_job.status in ("queued", "running"):
            return (gr.Textbox(value=_current_job.log_text), "⏳ A job is already running — wait for it to finish.",
                    gr.Timer(active=True), gr.DownloadButton(visible=False), gr.DownloadButton(visible=False),
                    *_empty_slots())
        if not url and not video_path:
            return (gr.Textbox(value=""), "⚠️ Paste a URL or upload a video first.", gr.Timer(active=False),
                    gr.DownloadButton(visible=False), gr.DownloadButton(visible=False), *_empty_slots())

        rules = pl.load_rules()
        rules["clips_per_video"] = min(int(rules["clips_per_video"]), MAX_SLOTS)
        if mode == "gemini" and not HAS_GEMINI:
            mode = "local"

        job = pl.Job()
        # keep uploads out of Gradio's cache so cleanup is ours
        if video_path:
            job_work = pl.WORK_DIR / job.id
            job_work.mkdir(parents=True, exist_ok=True)
            orig = Path(video_path)
            dst = job_work / (pl.sanitize_filename(orig.stem) + orig.suffix.lower())
            shutil.copy(video_path, dst)
            video_path = str(dst)
        if cookies_path:
            job_work = pl.WORK_DIR / job.id
            job_work.mkdir(parents=True, exist_ok=True)
            dst = job_work / "cookies.txt"
            shutil.copy(cookies_path, dst)
            cookies_path = str(dst)

        _current_job = job
        pl.run_in_background(job, url=url, local_file=video_path, cookies_path=cookies_path, mode=mode, rules=rules)

    return (gr.Textbox(value=job.log_text), f"🚀 Started job {job.id}", gr.Timer(active=True),
            gr.DownloadButton(visible=False), gr.DownloadButton(visible=False), *_empty_slots())


def poll_status():
    """Called by the timer every second while a job runs."""
    job = _current_job
    if job is None:
        return (gr.Textbox(value=""), "Idle", gr.Timer(active=False),
                gr.DownloadButton(visible=False), gr.DownloadButton(visible=False), *_empty_slots())
    pct = int(job.progress * 100)
    if job.status in ("queued", "running"):
        return (gr.Textbox(value=job.log_text), f"⏳ Running… {pct}%", gr.Timer(active=True),
                gr.DownloadButton(visible=False), gr.DownloadButton(visible=False), *_empty_slots())
    if job.status == "error":
        return (gr.Textbox(value=job.log_text), f"❌ Failed: {job.error}", gr.Timer(active=False),
                gr.DownloadButton(visible=False), gr.DownloadButton(visible=False), *_empty_slots())
    # done
    return (gr.Textbox(value=job.log_text), f"✅ Done — {len(job.clips)} clips", gr.Timer(active=False),
            gr.DownloadButton(value=job.zip_path, visible=bool(job.zip_path), label="⬇️ Download all (zip)"),
            gr.DownloadButton(value=job.captions_path, visible=bool(job.captions_path), label="⬇️ captions.md"),
            *_filled_slots(job.clips))


# --------------------------------------------------------------------------- #
# Rules tab
# --------------------------------------------------------------------------- #
def save_rules(text: str):
    try:
        merged = pl.save_rules_text(text)
        return f"✅ Saved. Effective: {merged['clips_per_video']} clips × {merged['min_len']}–{merged['max_len']}s"
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
                     len(h.get("clips", [])), h.get("source", "")[:80]])
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
    files = [str(out_dir / c["file"]) for c in entry.get("clips", []) if (out_dir / c["file"]).exists()]
    zp = out_dir / entry.get("zip", "clips.zip")
    md = f"### {entry.get('title','')}\n\nSource: {entry.get('source','')}\n\n"
    for c in entry.get("clips", []):
        md += f"- **{c['file']}** {c['start']:.1f}–{c['end']:.1f}s — {c.get('hook','')}\n"
    cap = out_dir / entry.get("captions", "captions.md")
    if cap.exists():
        md += "\n<details><summary>captions.md</summary>\n\n```\n" + cap.read_text(encoding="utf-8") + "\n```\n</details>"
    return (gr.File(value=files if files else None),
            gr.DownloadButton(value=str(zp) if zp.exists() else None, visible=zp.exists()), md)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Clipper") as demo:
    gr.Markdown("# ✂️ Clipper\nLong video → vertical clips with karaoke subtitles. CPU only, free tier friendly.")

    with gr.Tabs():
        # ---------------- Process ----------------
        with gr.Tab("Process"):
            with gr.Row():
                with gr.Column(scale=3):
                    url_in = gr.Textbox(label="YouTube / Kick / Google Drive link",
                                        placeholder="https://www.youtube.com/watch?v=...  or  https://kick.com/...  or  https://drive.google.com/file/d/.../view")
                    video_in = gr.File(label="… or upload a video file", file_types=["video", ".mp4", ".mkv", ".mov", ".webm"],
                                       file_count="single")
                with gr.Column(scale=2):
                    cookies_in = gr.File(label="cookies.txt (optional, Netscape format — bypasses YouTube bot checks)",
                                         file_types=[".txt"], file_count="single")
                    mode_in = gr.Radio(
                        choices=["local", "gemini"] if HAS_GEMINI else ["local"],
                        value="gemini" if HAS_GEMINI else "local",
                        label="Moment selection",
                        info=("local = free heuristic scoring; gemini = gemini-2.5-flash picks moments + writes hooks/captions"
                              if HAS_GEMINI else "Set GEMINI_API_KEY as a Space secret to enable Gemini selection"),
                    )
                    process_btn = gr.Button("▶️ Process", variant="primary", size="lg")
            status_md = gr.Markdown("Idle")
            log_box = gr.Textbox(label="Progress log", lines=12, max_lines=20, interactive=False, autoscroll=True)
            with gr.Row():
                zip_btn = gr.DownloadButton("⬇️ Download all (zip)", visible=False)
                captions_btn = gr.DownloadButton("⬇️ captions.md", visible=False)

            gr.Markdown("### Clips")
            slot_components = []
            for row in range((MAX_SLOTS + 3) // 4):
                with gr.Row():
                    for col in range(4):
                        with gr.Column():
                            v = gr.Video(visible=False, height=420, interactive=False)
                            m = gr.Markdown(visible=False)
                            d = gr.DownloadButton(visible=False, size="sm")
                            slot_components += [v, m, d]

            timer = gr.Timer(1.0, active=False)
            outputs = [log_box, status_md, timer, zip_btn, captions_btn, *slot_components]
            process_btn.click(start_processing, inputs=[url_in, video_in, cookies_in, mode_in], outputs=outputs)
            timer.tick(poll_status, outputs=outputs)

        # ---------------- Rules ----------------
        with gr.Tab("Rules"):
            gr.Markdown("Edit `rules.yaml`. Changes apply to the next job.")
            rules_box = gr.Code(value=pl.read_rules_text(), language="yaml", lines=28, label="rules.yaml")
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
            gr.Markdown("Last 20 processed videos. Output folders older than that are deleted automatically.")
            hist_refresh = gr.Button("🔄 Refresh")
            hist_table = gr.Dataframe(headers=["finished", "id", "title", "mode", "clips", "source"],
                                      value=history_table(), interactive=False, wrap=True)
            _choices = history_choices()
            hist_pick = gr.Dropdown(choices=_choices, value=_choices[0] if _choices else None, label="Open entry")
            hist_md = gr.Markdown("")
            hist_files = gr.File(label="Clips", file_count="multiple", interactive=False)
            hist_zip = gr.DownloadButton("⬇️ Download all (zip)", visible=False)
            hist_refresh.click(refresh_history, outputs=[hist_table, hist_pick])
            hist_pick.change(show_history_entry, inputs=[hist_pick], outputs=[hist_files, hist_zip, hist_md])

    # refresh history whenever the page loads
    demo.load(refresh_history, outputs=[hist_table, hist_pick])


if __name__ == "__main__":
    auth = (USERNAME, PASSWORD) if PASSWORD else None
    if auth is None:
        print("WARNING: PASSWORD env var not set — UI is unprotected!", flush=True)
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        auth=auth,
        auth_message="Clipper — enter the password from the Space secrets" if auth else None,
        allowed_paths=[str(pl.OUT_DIR)],
        max_file_size="4gb",
        show_error=True,
    )
