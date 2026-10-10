"""
Clipper — mobile-first single-user Gradio UI.

Main screen is one vertical flow: 1) video, 2) campaign, 3) one big button. Results come back as cards.
The job runs inside the HTTP request of the button click (streaming generator), which keeps Cloud Run's
CPU allocated for the whole job without paying for always-on CPU.
"""
from __future__ import annotations

import base64
import os
import shutil
import threading
from pathlib import Path

import gradio as gr
import yaml

import gcs
import i18n
import pipeline as pl
import store

MAX_SLOTS = 8
PASSWORD = os.environ.get("PASSWORD", "").strip()
USERNAME = os.environ.get("USERNAME", "admin").strip() or "admin"
HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY", "").strip())
ON_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))
MAX_UPLOAD = os.environ.get("MAX_UPLOAD", "2gb")

LEN_PRESETS = [("15–30 s", "15-30"), ("20–45 s", "20-45"), ("30–60 s", "30-60"), ("10–20 s", "10-20")]
POSITIONS = ["bottom_right", "bottom_left", "top_right", "top_left"]

pl.OUT_DIR.mkdir(parents=True, exist_ok=True)
pl.WORK_DIR.mkdir(parents=True, exist_ok=True)
pl.LOGOS_DIR.mkdir(parents=True, exist_ok=True)
store.sync_down()

_state_lock = threading.Lock()
_current_job: pl.Job | None = None
_i18n_registry: list[tuple] = []        # (component, key, kind)


def LANG() -> str:
    return store.load_settings()["language"]


def T(key: str, **fmt) -> str:
    return i18n.t(LANG(), key, **fmt)


def reg(component, key: str, kind: str = "label"):
    """Register a component for live re-labelling on language switch."""
    _i18n_registry.append((component, key, kind))
    return component


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _file_path(f) -> str | None:
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get("path") or f.get("name")
    return getattr(f, "path", None) or getattr(f, "name", None)


def _progress_html(pct: int, error: bool = False) -> str:
    color = "#ef4444" if error else "#f59e0b"
    return (f'<div style="background:#2a2a2a;border-radius:10px;height:16px;overflow:hidden">'
            f'<div style="width:{max(2, min(100, pct))}%;background:{color};height:16px;transition:width .4s"></div></div>')


def _campaign_choices():
    return [(T("campaign_none"), "")] + [(c["name"], c["name"]) for c in store.load_campaigns()]


def _logo_choices(extra: str = ""):
    names = pl.list_saved_logos()
    if extra and extra not in names:
        names.append(extra)
    return [("—", "")] + [(n, n) for n in names]


def _cards(clips: list[dict], lang: str):
    """Updates for the MAX_SLOTS result cards: (column, video, caption textbox, download button) each."""
    out = []
    for i in range(MAX_SLOTS):
        if i < len(clips):
            c = clips[i]
            name = Path(c["file"]).name
            out += [gr.Column(visible=True),
                    gr.Video(value=c["file"], label=f"{name} · {c['end'] - c['start']:.0f}s"),
                    gr.Textbox(value=c.get("caption", "").rstrip(), label=i18n.t(lang, "caption")),
                    gr.DownloadButton(value=c["file"], label=f"{i18n.t(lang, 'download')} {name}")]
        else:
            out += [gr.Column(visible=False), gr.Video(value=None), gr.Textbox(value=""), gr.DownloadButton(value=None)]
    return out


def _links_md(job_id: str, title: str = "") -> str:
    if not gcs.enabled():
        return ""
    zip_url = gcs.signed_url(gcs.blob_name(job_id, "clips.zip"),
                             download_name=f"{pl.sanitize_filename(title or job_id)}_clips.zip")
    cap_url = gcs.signed_url(gcs.blob_name(job_id, "captions.md"), download_name="captions.md")
    parts = []
    if zip_url:
        parts.append(f"[clips.zip]({zip_url})")
    if cap_url:
        parts.append(f"[captions.md]({cap_url})")
    return (f"☁️ {T('cloud_links')}: " + " · ".join(parts)) if parts else ""


def _frame(job: pl.Job | None, lang: str, status: str | None = None, final: bool = False):
    """One UI update: stage text, progress bar, log, zip button, cloud links, cards."""
    if job is None:
        return (status or i18n.t(lang, "idle"), _progress_html(0), "",
                gr.DownloadButton(visible=False), gr.Markdown(value="", visible=False), *_cards([], lang))
    pct = int(job.progress * 100)
    if job.status in ("queued", "running"):
        stage = status or i18n.stage_text(lang, job.stage, job.progress)
        return (f"⏳ {stage}", _progress_html(pct), job.log_text,
                gr.DownloadButton(visible=False), gr.Markdown(value="", visible=False), *_cards([], lang))
    if job.status == "error":
        return (f"❌ {i18n.friendly_error(job.error or '', lang)}", _progress_html(pct, error=True), job.log_text,
                gr.DownloadButton(visible=False), gr.Markdown(value="", visible=False), *_cards([], lang))
    links = _links_md(job.id, job.title) if final else ""
    return (i18n.t(lang, "done", n=len(job.clips)), _progress_html(100), job.log_text,
            gr.DownloadButton(value=job.zip_path, visible=bool(job.zip_path), label=i18n.t(lang, "download_all")),
            gr.Markdown(value=links, visible=bool(links)), *_cards(job.clips, lang))


# --------------------------------------------------------------------------- #
# main action
# --------------------------------------------------------------------------- #
def run_processing(url: str, video_file, campaign_name: str, cookies_file, mode: str, whisper: str, len_preset: str):
    global _current_job
    lang = LANG()
    url = (url or "").strip()
    video_path = _file_path(video_file)
    cookies_path = _file_path(cookies_file)

    with _state_lock:
        if _current_job is not None and _current_job.status in ("queued", "running"):
            yield _frame(_current_job, lang, status=i18n.t(lang, "busy"))
            return
        if not url and not video_path:
            yield _frame(None, lang, status="⚠️ " + i18n.t(lang, "need_input"))
            return

        rules = pl.load_rules()
        # advanced
        if whisper in pl.ALLOWED_WHISPER_MODELS:
            rules["whisper_model"] = whisper
        if len_preset and "-" in len_preset:
            lo, hi = len_preset.split("-")
            rules["min_len"], rules["max_len"] = int(lo), int(hi)
        if mode == "gemini" and not HAS_GEMINI:
            mode = "local"
        # campaign
        logo_overrides: dict = {}
        camp = store.get_campaign(campaign_name) if campaign_name else None
        if camp:
            rules["clips_per_video"] = int(camp.get("clips_per_video") or rules["clips_per_video"])
            if camp.get("caption_template"):
                rules["caption_template"] = camp["caption_template"]
            rules["hashtags"] = camp.get("hashtags", "")
            kw = camp.get("keywords_boost") or []
            rules["keywords_boost"] = list(dict.fromkeys(list(rules.get("keywords_boost") or []) + list(kw)))
            logo_overrides["campaign_logo"] = camp.get("campaign_logo") or None
        rules["clips_per_video"] = min(int(rules["clips_per_video"]), MAX_SLOTS)

        job = pl.Job()
        job.campaign = camp["name"] if camp else ""
        job_work = pl.WORK_DIR / job.id
        if video_path:
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
        job.set_stage("st_start", progress=0.01)
        _current_job = job

    worker = pl.run_in_background(job, url=url, local_file=video_path, cookies_path=cookies_path,
                                  mode=mode, rules=rules, logo_overrides=logo_overrides)
    yield _frame(job, lang)
    last = None
    while worker.is_alive():
        worker.join(timeout=1.0)
        snap = (len(job.log_lines), job.stage, round(job.progress, 2))
        if snap != last:
            last = snap
            yield _frame(job, lang)
    yield _frame(job, lang, final=True)


def attach_running():
    lang = LANG()
    job = _current_job
    if job is None:
        return _frame(None, lang)
    return _frame(job, lang, final=(job.status == "done"))


# --------------------------------------------------------------------------- #
# campaigns
# --------------------------------------------------------------------------- #
def open_campaign_form(selected: str):
    c = store.get_campaign(selected) if selected else None
    c = c or dict(store.DEFAULT_CAMPAIGN)
    return (gr.Column(visible=True), c["name"], gr.Dropdown(choices=_logo_choices(c.get("campaign_logo") or ""), value=c.get("campaign_logo") or ""),
            None, c.get("hashtags", ""), c.get("caption_template", ""), int(c.get("clips_per_video") or 4),
            ", ".join(c.get("keywords_boost") or []), "")


def close_campaign_form():
    return gr.Column(visible=False), ""


def save_campaign(name: str, logo_pick: str, logo_upload, hashtags: str, caption_template: str, clips: float,
                  keywords: str):
    name = (name or "").strip()
    if not name:
        return gr.Dropdown(choices=_campaign_choices()), gr.Column(visible=True), "⚠️ " + T("camp_need_name")
    logo_name = logo_pick or ""
    up = _file_path(logo_upload)
    if up:
        logo_name = pl.save_logo(up, Path(up).stem)
    store.upsert_campaign({
        "name": name, "campaign_logo": logo_name, "hashtags": (hashtags or "").strip(),
        "caption_template": caption_template or store.DEFAULT_CAMPAIGN["caption_template"],
        "clips_per_video": int(clips or 4),
        "keywords_boost": [k.strip() for k in (keywords or "").split(",") if k.strip()],
    })
    return gr.Dropdown(choices=_campaign_choices(), value=name), gr.Column(visible=False), "✅ " + T("camp_saved")


def delete_campaign(name: str):
    if name:
        store.delete_campaign(name.strip())
    return gr.Dropdown(choices=_campaign_choices(), value=""), gr.Column(visible=False), "🗑️ " + T("camp_deleted")


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def settings_values():
    r = pl.load_rules()
    my = r["logos"]["my_logo"]
    sub = r["subtitle"]
    return (bool(my.get("enabled")), my.get("file") or "",
            my.get("position", "bottom_right"), int(sub.get("size", 72)), sub.get("color", "#FFFFFF"),
            sub.get("highlight_color", "#FFD400"), sub.get("position", "lower_third"), int(sub.get("words_per_line", 3)),
            r.get("hook_style", "text_overlay") == "text_overlay")


def save_settings(my_enabled: bool, my_pick: str, my_upload, my_pos: str, size: float, color: str, hi: str,
                  sub_pos: str, wpl: float, hook_on: bool):
    r = pl.load_rules()
    up = _file_path(my_upload)
    logo_name = my_pick or r["logos"]["my_logo"].get("file", "")
    if up:
        logo_name = pl.save_logo(up, Path(up).stem)
    r["logos"]["my_logo"].update({"enabled": bool(my_enabled and logo_name), "file": logo_name, "position": my_pos})
    # keep the campaign logo out of the same corner
    if r["logos"]["campaign_logo"].get("position") == my_pos:
        r["logos"]["campaign_logo"]["position"] = "top_left" if my_pos != "top_left" else "top_right"
    r["subtitle"].update({"size": int(size), "color": color, "highlight_color": hi, "position": sub_pos,
                          "words_per_line": int(wpl)})
    r["hook_style"] = "text_overlay" if hook_on else "none"
    text = yaml.safe_dump(r, sort_keys=False, allow_unicode=True)
    pl.save_rules_text(text)
    store.write_yaml(store.RULES_PATH, None, text=text)
    return "✅ " + T("settings_saved"), gr.Dropdown(choices=_logo_choices(logo_name), value=logo_name), pl.read_rules_text()


def save_rules_text(text: str):
    try:
        pl.save_rules_text(text)
        store.write_yaml(store.RULES_PATH, None, text=text)
        return "✅ " + T("rules_saved")
    except Exception as e:  # noqa: BLE001
        return "❌ " + T("rules_bad", e=e)


def toggle_expert(on: bool):
    store.save_settings(expert_mode=bool(on))
    return gr.Column(visible=bool(on)), pl.read_rules_text()


def switch_language(lang: str):
    store.save_settings(language=lang)
    updates = []
    for comp, key, kind in _i18n_registry:
        text = i18n.t(lang, key)
        if kind == "label":
            updates.append(gr.update(label=text))
        elif kind == "value":
            updates.append(gr.update(value=text))
        elif kind == "campaign_choices":
            updates.append(gr.Dropdown(choices=[(i18n.t(lang, "campaign_none"), "")] +
                                       [(c["name"], c["name"]) for c in store.load_campaigns()],
                                       label=i18n.t(lang, "campaign_label")))
        elif kind == "mode_choices":
            ch = [(i18n.t(lang, "mode_local"), "local")] + ([(i18n.t(lang, "mode_gemini"), "gemini")] if HAS_GEMINI else [])
            updates.append(gr.Radio(choices=ch, label=i18n.t(lang, "mode_label")))
    updates.append(history_html(lang))
    return updates


# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #
def _img_src(local: Path, job_id: str, name: str) -> str | None:
    if local.exists():
        try:
            b = local.read_bytes()
            return "data:image/jpeg;base64," + base64.b64encode(b).decode()
        except Exception:  # noqa: BLE001
            return None
    if gcs.enabled():
        return gcs.signed_url(gcs.blob_name(job_id, name), minutes=60)
    return None


def history_html(lang: str | None = None) -> str:
    lang = lang or LANG()
    entries = pl.load_history()
    if not entries:
        return f"<p style='opacity:.7'>{i18n.t(lang, 'hist_empty')}</p>"
    rows = []
    for h in entries[:20]:
        job_id = h.get("id", "")
        out_dir = pl.OUT_DIR / job_id
        thumbs = []
        for c in h.get("clips", [])[:4]:
            tn = c.get("thumb") or ""
            src = _img_src(out_dir / tn, job_id, tn) if tn else None
            if src:
                thumbs.append(f'<img src="{src}" style="width:23%;aspect-ratio:9/16;object-fit:cover;border-radius:8px;background:#222">')
        zp = out_dir / h.get("zip", "clips.zip")
        link = None
        if zp.exists():
            link = f"/gradio_api/file={zp}"
        elif h.get("gcs", {}).get("zip"):
            link = gcs.signed_url(gcs.blob_name(job_id, "clips.zip"), download_name=f"{pl.sanitize_filename(h.get('title') or job_id)}_clips.zip")
        btn = (f'<a href="{link}" style="display:inline-block;padding:10px 18px;background:#f59e0b;color:#111;'
               f'border-radius:10px;font-weight:600;text-decoration:none">⬇️ {i18n.t(lang, "hist_zip")}</a>'
               if link else f'<span style="opacity:.6">{i18n.t(lang, "hist_no_files")}</span>')
        label = h.get("campaign") or h.get("title", "")
        rows.append(
            f'<div style="padding:12px 0;border-bottom:1px solid #333">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">'
            f'<div><b>{h.get("finished", "")}</b> · {label}</div>{btn}</div>'
            f'<div style="display:flex;gap:2%;margin-top:8px">{"".join(thumbs)}</div></div>'
        )
    return "".join(rows)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container {max-width: 720px !important; margin: 0 auto !important;}
#go-btn {min-height: 64px; font-size: 1.25rem; font-weight: 700;}
button.lg, .lg {min-height: 52px;}
.card video {max-height: 60vh;}
footer {display: none !important;}
"""
FORCE_DARK_JS = """
() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') { u.searchParams.set('__theme', 'dark'); window.location.replace(u); }
}
"""

_settings = store.load_settings()
_lang0 = _settings["language"]

with gr.Blocks(title="Clipper") as demo:
    reg(gr.Markdown(i18n.t(_lang0, "title")), "title", "value")
    reg(gr.Markdown(i18n.t(_lang0, "subtitle")), "subtitle", "value")

    with gr.Tabs():
        # ================= main =================
        with gr.Tab(i18n.t(_lang0, "tab_main")) as tab_main:
            reg(tab_main, "tab_main", "label")
            reg(gr.Markdown(i18n.t(_lang0, "step1")), "step1", "value")
            url_in = reg(gr.Textbox(label=i18n.t(_lang0, "url_label"), placeholder=i18n.t(_lang0, "url_ph"), lines=1), "url_label")
            video_in = reg(gr.File(label=i18n.t(_lang0, "upload_label"), file_types=["video", ".mp4", ".mkv", ".mov", ".webm"],
                                   file_count="single", height=110), "upload_label")

            reg(gr.Markdown(i18n.t(_lang0, "step2")), "step2", "value")
            campaign_in = reg(gr.Dropdown(choices=_campaign_choices(), value="", label=i18n.t(_lang0, "campaign_label")),
                              "campaign_label", "campaign_choices")
            new_camp_btn = reg(gr.Button(i18n.t(_lang0, "new_campaign"), size="lg"), "new_campaign", "value")
            with gr.Column(visible=False) as camp_form:
                reg(gr.Markdown(i18n.t(_lang0, "camp_form_title")), "camp_form_title", "value")
                camp_name = reg(gr.Textbox(label=i18n.t(_lang0, "camp_name")), "camp_name")
                camp_logo_pick = reg(gr.Dropdown(choices=_logo_choices(), value="", allow_custom_value=True,
                                                 label=i18n.t(_lang0, "camp_logo_pick")), "camp_logo_pick")
                camp_logo_up = reg(gr.File(label=i18n.t(_lang0, "camp_logo_upload"), file_types=[".png", ".webp"],
                                           file_count="single", height=90), "camp_logo_upload")
                camp_hashtags = reg(gr.Textbox(label=i18n.t(_lang0, "camp_hashtags"), value="#shorts #viral"), "camp_hashtags")
                camp_caption = reg(gr.Textbox(label=i18n.t(_lang0, "camp_caption"), lines=4,
                                              value=store.DEFAULT_CAMPAIGN["caption_template"]), "camp_caption")
                camp_clips = reg(gr.Slider(1, MAX_SLOTS, value=4, step=1, label=i18n.t(_lang0, "camp_clips")), "camp_clips")
                camp_keywords = reg(gr.Textbox(label=i18n.t(_lang0, "camp_keywords")), "camp_keywords")
                with gr.Row():
                    camp_save_btn = reg(gr.Button(i18n.t(_lang0, "camp_save"), variant="primary", size="lg"), "camp_save", "value")
                    camp_close_btn = reg(gr.Button(i18n.t(_lang0, "camp_cancel"), size="lg"), "camp_cancel", "value")
                camp_del_btn = reg(gr.Button(i18n.t(_lang0, "camp_delete"), variant="stop", size="sm"), "camp_delete", "value")
            camp_status = gr.Markdown("")

            reg(gr.Markdown(i18n.t(_lang0, "step3")), "step3", "value")
            go_btn = reg(gr.Button(i18n.t(_lang0, "go"), variant="primary", size="lg", elem_id="go-btn"), "go", "value")
            reg(gr.Markdown(f"<small>{i18n.t(_lang0, 'keep_open')}</small>"), "keep_open", "value")

            with gr.Accordion(i18n.t(_lang0, "advanced"), open=False) as adv_acc:
                reg(adv_acc, "advanced", "label")
                cookies_in = reg(gr.File(label=i18n.t(_lang0, "cookies_label"), file_types=[".txt"], file_count="single", height=90),
                                 "cookies_label")
                mode_choices = [(i18n.t(_lang0, "mode_local"), "local")] + ([(i18n.t(_lang0, "mode_gemini"), "gemini")] if HAS_GEMINI else [])
                mode_in = reg(gr.Radio(choices=mode_choices, value="gemini" if HAS_GEMINI else "local",
                                       label=i18n.t(_lang0, "mode_label")), "mode_label", "mode_choices")
                whisper_in = reg(gr.Radio(choices=["base", "small"], value=pl.load_rules().get("whisper_model", "base"),
                                          label=i18n.t(_lang0, "whisper_label")), "whisper_label")
                len_in = reg(gr.Dropdown(choices=LEN_PRESETS, value="15-30", label=i18n.t(_lang0, "len_label")), "len_label")

            stage_md = gr.Markdown(i18n.t(_lang0, "idle"))
            progress_html = gr.HTML(_progress_html(0))
            with gr.Accordion(i18n.t(_lang0, "details"), open=False) as det_acc:
                reg(det_acc, "details", "label")
                log_box = gr.Textbox(label="log", lines=10, max_lines=16, interactive=False, autoscroll=True, show_label=False)

            reg(gr.Markdown(i18n.t(_lang0, "results")), "results", "value")
            zip_btn = gr.DownloadButton(i18n.t(_lang0, "download_all"), visible=False, size="lg", variant="primary")
            links_md = gr.Markdown("", visible=False)
            card_components = []
            for _row in range((MAX_SLOTS + 1) // 2):
                with gr.Row():
                    for _col in range(2):
                        with gr.Column(visible=False, elem_classes=["card"]) as col:
                            v = gr.Video(interactive=False, height=480)
                            cap = gr.Textbox(label=i18n.t(_lang0, "caption"), lines=4, buttons=["copy"], interactive=False)
                            d = gr.DownloadButton(i18n.t(_lang0, "download"), size="lg")
                            card_components += [col, v, cap, d]

            main_outputs = [stage_md, progress_html, log_box, zip_btn, links_md, *card_components]
            go_btn.click(run_processing,
                         inputs=[url_in, video_in, campaign_in, cookies_in, mode_in, whisper_in, len_in],
                         outputs=main_outputs, concurrency_limit=1, concurrency_id="process")

            camp_form_outputs = [camp_form, camp_name, camp_logo_pick, camp_logo_up, camp_hashtags, camp_caption,
                                 camp_clips, camp_keywords, camp_status]
            new_camp_btn.click(open_campaign_form, inputs=[campaign_in], outputs=camp_form_outputs)
            camp_close_btn.click(close_campaign_form, outputs=[camp_form, camp_status])
            camp_save_btn.click(save_campaign,
                                inputs=[camp_name, camp_logo_pick, camp_logo_up, camp_hashtags, camp_caption, camp_clips, camp_keywords],
                                outputs=[campaign_in, camp_form, camp_status])
            camp_del_btn.click(delete_campaign, inputs=[camp_name], outputs=[campaign_in, camp_form, camp_status])

        # ================= settings =================
        with gr.Tab(i18n.t(_lang0, "tab_settings")) as tab_settings:
            reg(tab_settings, "tab_settings", "label")
            lang_in = reg(gr.Radio(choices=[("Latviski", "lv"), ("English", "en")], value=_lang0,
                                   label=i18n.t(_lang0, "settings_lang")), "settings_lang")
            reg(gr.Markdown(i18n.t(_lang0, "settings_mylogo")), "settings_mylogo", "value")
            s0 = settings_values()
            my_enabled = reg(gr.Checkbox(value=s0[0], label=i18n.t(_lang0, "mylogo_enabled")), "mylogo_enabled")
            my_pick = reg(gr.Dropdown(choices=_logo_choices(s0[1]), value=s0[1], allow_custom_value=True,
                                      label=i18n.t(_lang0, "mylogo_pick")), "mylogo_pick")
            my_upload = reg(gr.File(label=i18n.t(_lang0, "mylogo_upload"), file_types=[".png", ".webp"], file_count="single", height=90),
                            "mylogo_upload")
            my_pos = reg(gr.Dropdown(choices=POSITIONS, value=s0[2], label=i18n.t(_lang0, "mylogo_pos")), "mylogo_pos")
            reg(gr.Markdown(i18n.t(_lang0, "settings_sub")), "settings_sub", "value")
            sub_size = reg(gr.Slider(40, 110, value=s0[3], step=2, label=i18n.t(_lang0, "sub_size")), "sub_size")
            sub_color = reg(gr.ColorPicker(value=s0[4], label=i18n.t(_lang0, "sub_color")), "sub_color")
            sub_hi = reg(gr.ColorPicker(value=s0[5], label=i18n.t(_lang0, "sub_hi")), "sub_hi")
            sub_pos = reg(gr.Dropdown(choices=["lower_third", "center", "top"], value=s0[6], label=i18n.t(_lang0, "sub_pos")), "sub_pos")
            sub_wpl = reg(gr.Slider(1, 6, value=s0[7], step=1, label=i18n.t(_lang0, "sub_wpl")), "sub_wpl")
            hook_on = reg(gr.Checkbox(value=s0[8], label=i18n.t(_lang0, "hook_on")), "hook_on")
            settings_save_btn = reg(gr.Button(i18n.t(_lang0, "settings_save"), variant="primary", size="lg"), "settings_save", "value")
            settings_status = gr.Markdown("")
            expert_in = reg(gr.Checkbox(value=bool(_settings.get("expert_mode")), label=i18n.t(_lang0, "expert")), "expert")
            with gr.Column(visible=bool(_settings.get("expert_mode"))) as expert_col:
                rules_box = gr.Code(value=pl.read_rules_text(), language="yaml", lines=28, label="rules.yaml")
                with gr.Row():
                    rules_save_btn = reg(gr.Button(i18n.t(_lang0, "rules_save"), variant="primary"), "rules_save", "value")
                    rules_reload_btn = reg(gr.Button(i18n.t(_lang0, "rules_reload")), "rules_reload", "value")
                rules_status = gr.Markdown("")

            settings_save_btn.click(save_settings,
                                    inputs=[my_enabled, my_pick, my_upload, my_pos, sub_size, sub_color, sub_hi, sub_pos, sub_wpl, hook_on],
                                    outputs=[settings_status, my_pick, rules_box])
            expert_in.change(toggle_expert, inputs=[expert_in], outputs=[expert_col, rules_box])
            rules_save_btn.click(save_rules_text, inputs=[rules_box], outputs=[rules_status])
            rules_reload_btn.click(lambda: pl.read_rules_text(), outputs=[rules_box])

        # ================= history =================
        with gr.Tab(i18n.t(_lang0, "tab_history")) as tab_history:
            reg(tab_history, "tab_history", "label")
            hist_refresh = reg(gr.Button(i18n.t(_lang0, "hist_refresh"), size="lg"), "hist_refresh", "value")
            hist_html = gr.HTML(history_html(_lang0))
            hist_refresh.click(lambda: history_html(), outputs=[hist_html])

    # language switch re-labels every registered component (+ history html)
    i18n_outputs = [c for c, _, _ in _i18n_registry] + [hist_html]

    def _switch(lang):
        return switch_language(lang)

    lang_in.change(_switch, inputs=[lang_in], outputs=i18n_outputs)

    demo.load(attach_running, outputs=main_outputs)
    demo.load(lambda: history_html(), outputs=[hist_html])



if __name__ == "__main__":
    auth = (USERNAME, PASSWORD) if PASSWORD else None
    if auth is None:
        print("WARNING: PASSWORD env var not set — UI is unprotected!", flush=True)
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        auth=auth,
        auth_message="Clipper" if auth else None,
        allowed_paths=[str(pl.OUT_DIR)],
        max_file_size="2gb",
        show_error=True,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="zinc"),
        css=CSS,
        js=FORCE_DARK_JS,
    )
