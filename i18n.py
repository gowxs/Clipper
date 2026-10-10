"""UI strings (Latvian default, English) and human-readable error / progress messages."""
from __future__ import annotations

T: dict[str, dict[str, str]] = {
    "lv": {
        "title": "# ✂️ Clipper",
        "subtitle": "No gara video līdz vertikāliem klipiem ar subtitriem. Trīs soļi.",
        "tab_main": "Klipi",
        "tab_settings": "Iestatījumi",
        "tab_history": "Vēsture",
        "step1": "### 1 · Video",
        "url_label": "Ielīmē saiti (YouTube / Kick / Google Drive / gs://)",
        "url_ph": "https://… vai gs://…",
        "or_upload": "vai augšupielādē failu",
        "upload_label": "Augšupielādēt failu",
        "step2": "### 2 · Kampaņa",
        "campaign_label": "Kampaņas profils",
        "campaign_none": "(bez kampaņas)",
        "new_campaign": "+ Jauna kampaņa",
        "camp_form_title": "**Jauna / rediģēt kampaņu**",
        "camp_name": "Nosaukums",
        "camp_logo_pick": "Kampaņas logo (saglabātais)",
        "camp_logo_upload": "vai augšupielādē logo (PNG)",
        "camp_hashtags": "Hashtagi",
        "camp_caption": "Paraksta veidne ({hook} {title} {hashtags})",
        "camp_clips": "Klipu skaits",
        "camp_keywords": "Atslēgvārdi, kas jāizceļ (ar komatu)",
        "camp_save": "Saglabāt kampaņu",
        "camp_delete": "Dzēst šo kampaņu",
        "camp_cancel": "Aizvērt",
        "camp_saved": "Kampaņa saglabāta.",
        "camp_deleted": "Kampaņa dzēsta.",
        "camp_need_name": "Ievadi kampaņas nosaukumu.",
        "step3": "### 3 · Gatavs",
        "go": "🎬 Veidot klipus",
        "advanced": "Papildu",
        "cookies_label": "cookies.txt (ja YouTube bloķē)",
        "mode_label": "Momentu izvēle",
        "mode_local": "local (bez maksas)",
        "mode_gemini": "gemini (AI izvēlas un raksta parakstus)",
        "whisper_label": "Transkripcijas modelis",
        "len_label": "Klipa garums (s)",
        "idle": "Gatavs darbam.",
        "details": "Rādīt detaļas",
        "results": "### Klipi",
        "download_all": "⬇️ Lejupielādēt visus (zip)",
        "download": "⬇️ Lejupielādēt",
        "caption": "Paraksts",
        "cloud_links": "Saites mākonī",
        "need_input": "Ielīmē saiti vai augšupielādē failu.",
        "busy": "Jau notiek apstrāde — pagaidi, līdz tā beidzas.",
        "keep_open": "Turi šo cilni atvērtu, kamēr notiek apstrāde.",
        "done": "✅ Gatavs — {n} klipi",
        "failed": "❌ Neizdevās",
        # settings
        "settings_lang": "Valoda / Language",
        "settings_mylogo": "### Mans logo",
        "mylogo_enabled": "Rādīt manu logo uz visiem klipiem",
        "mylogo_pick": "Logo fails (saglabātais)",
        "mylogo_upload": "vai augšupielādē (PNG ar caurspīdīgumu)",
        "mylogo_pos": "Stūris",
        "settings_sub": "### Subtitru stils",
        "sub_size": "Burtu izmērs",
        "sub_color": "Teksta krāsa",
        "sub_hi": "Aktīvā vārda krāsa",
        "sub_pos": "Novietojums",
        "sub_wpl": "Vārdi rindā",
        "hook_on": "Rādīt hook tekstu pirmās 3 s",
        "settings_save": "💾 Saglabāt iestatījumus",
        "settings_saved": "Iestatījumi saglabāti.",
        "expert": "Eksperts: rādīt rules.yaml",
        "rules_save": "Saglabāt rules.yaml",
        "rules_reload": "Pārlādēt",
        "rules_saved": "rules.yaml saglabāts.",
        "rules_bad": "Nav saglabāts — kļūda: {e}",
        # history
        "hist_empty": "Vēl nav apstrādātu video.",
        "hist_refresh": "🔄 Atsvaidzināt",
        "hist_zip": "zip",
        "hist_no_files": "faili vairs nav pieejami",
        # progress stages
        "st_start": "Sākam…",
        "st_download": "Lejupielādē video…",
        "st_upload_src": "Sagatavo augšupielādēto failu…",
        "st_trim": "Saīsina garo video…",
        "st_audio": "Atdala audio…",
        "st_transcribe": "Transkribē runu… ({pct}%)",
        "st_select": "Izvēlas labākos momentus…",
        "st_render": "Veido klipu {i}/{n}…",
        "st_uploadgcs": "Saglabā mākonī…",
        "st_done": "Gatavs!",
        "st_error": "Kļūda",
    },
    "en": {
        "title": "# ✂️ Clipper",
        "subtitle": "Long video → vertical captioned clips. Three steps.",
        "tab_main": "Clips",
        "tab_settings": "Settings",
        "tab_history": "History",
        "step1": "### 1 · Video",
        "url_label": "Paste a link (YouTube / Kick / Google Drive / gs://)",
        "url_ph": "https://… or gs://…",
        "or_upload": "or upload a file",
        "upload_label": "Upload a file",
        "step2": "### 2 · Campaign",
        "campaign_label": "Campaign profile",
        "campaign_none": "(no campaign)",
        "new_campaign": "+ New campaign",
        "camp_form_title": "**New / edit campaign**",
        "camp_name": "Name",
        "camp_logo_pick": "Campaign logo (saved)",
        "camp_logo_upload": "or upload a logo (PNG)",
        "camp_hashtags": "Hashtags",
        "camp_caption": "Caption template ({hook} {title} {hashtags})",
        "camp_clips": "Number of clips",
        "camp_keywords": "Keywords to boost (comma separated)",
        "camp_save": "Save campaign",
        "camp_delete": "Delete this campaign",
        "camp_cancel": "Close",
        "camp_saved": "Campaign saved.",
        "camp_deleted": "Campaign deleted.",
        "camp_need_name": "Enter a campaign name.",
        "step3": "### 3 · Go",
        "go": "🎬 Make clips",
        "advanced": "Advanced",
        "cookies_label": "cookies.txt (if YouTube blocks)",
        "mode_label": "Moment selection",
        "mode_local": "local (free)",
        "mode_gemini": "gemini (AI picks moments and writes captions)",
        "whisper_label": "Transcription model",
        "len_label": "Clip length (s)",
        "idle": "Ready.",
        "details": "Show details",
        "results": "### Clips",
        "download_all": "⬇️ Download all (zip)",
        "download": "⬇️ Download",
        "caption": "Caption",
        "cloud_links": "Cloud links",
        "need_input": "Paste a link or upload a file.",
        "busy": "A job is already running — wait for it to finish.",
        "keep_open": "Keep this tab open while processing.",
        "done": "✅ Done — {n} clips",
        "failed": "❌ Failed",
        "settings_lang": "Language / Valoda",
        "settings_mylogo": "### My logo",
        "mylogo_enabled": "Show my logo on every clip",
        "mylogo_pick": "Logo file (saved)",
        "mylogo_upload": "or upload (PNG with transparency)",
        "mylogo_pos": "Corner",
        "settings_sub": "### Subtitle style",
        "sub_size": "Font size",
        "sub_color": "Text colour",
        "sub_hi": "Active word colour",
        "sub_pos": "Position",
        "sub_wpl": "Words per line",
        "hook_on": "Show hook text for the first 3 s",
        "settings_save": "💾 Save settings",
        "settings_saved": "Settings saved.",
        "expert": "Expert: show rules.yaml",
        "rules_save": "Save rules.yaml",
        "rules_reload": "Reload",
        "rules_saved": "rules.yaml saved.",
        "rules_bad": "Not saved — error: {e}",
        "hist_empty": "No processed videos yet.",
        "hist_refresh": "🔄 Refresh",
        "hist_zip": "zip",
        "hist_no_files": "files no longer available",
        "st_start": "Starting…",
        "st_download": "Downloading video…",
        "st_upload_src": "Preparing uploaded file…",
        "st_trim": "Trimming long video…",
        "st_audio": "Extracting audio…",
        "st_transcribe": "Transcribing speech… ({pct}%)",
        "st_select": "Picking the best moments…",
        "st_render": "Rendering clip {i}/{n}…",
        "st_uploadgcs": "Saving to cloud…",
        "st_done": "Done!",
        "st_error": "Error",
    },
}

ERRORS: dict[str, dict[str, str]] = {
    "lv": {
        "yt_bot": "YouTube bloķē lejupielādi no servera — pievieno cookies.txt sadaļā “Papildu” vai augšupielādē failu.",
        "yt_private": "Video nav pieejams (privāts, dzēsts vai ar vecuma ierobežojumu). Pamēģini ar cookies.txt.",
        "drive_private": "Google Drive fails nav publisks — kopīgo kā “Ikviens ar saiti” un ielīmē saiti vēlreiz.",
        "drive_quota": "Google Drive pārsniedza lejupielādes kvotu šim failam — pamēģini vēlāk vai augšupielādē failu.",
        "no_audio": "Video nav audio celiņa — klipus bez runas izveidot nevar.",
        "no_input": "Ielīmē saiti vai augšupielādē failu.",
        "too_big": "Fails pārāk liels augšupielādei — izmanto saiti (YouTube / Drive) vai gs:// ceļu.",
        "net": "Tīkla kļūda lejupielādējot — pamēģini vēlreiz.",
        "gemini": "Gemini neatbildēja — klipi izvēlēti ar vietējo metodi.",
        "ffmpeg": "Video apstrāde (ffmpeg) neizdevās — iespējams, bojāts fails. Pamēģini citu failu.",
        "unknown": "Kaut kas nogāja greizi: {msg}",
    },
    "en": {
        "yt_bot": "YouTube blocks downloads from this server — add cookies.txt under “Advanced” or upload the file.",
        "yt_private": "Video unavailable (private, removed or age-restricted). Try with cookies.txt.",
        "drive_private": "The Google Drive file is not public — share it as “Anyone with the link” and paste the link again.",
        "drive_quota": "Google Drive download quota exceeded for this file — try later or upload the file.",
        "no_audio": "The video has no audio track — clips need speech.",
        "no_input": "Paste a link or upload a file.",
        "too_big": "File too large to upload — use a link (YouTube / Drive) or a gs:// path.",
        "net": "Network error while downloading — try again.",
        "gemini": "Gemini did not answer — clips were picked with the local method.",
        "ffmpeg": "Video processing (ffmpeg) failed — the file may be corrupt. Try another file.",
        "unknown": "Something went wrong: {msg}",
    },
}


def t(lang: str, key: str, **fmt) -> str:
    s = T.get(lang, T["lv"]).get(key) or T["en"].get(key, key)
    return s.format(**fmt) if fmt else s


def friendly_error(msg: str, lang: str) -> str:
    m = (msg or "").lower()
    e = ERRORS.get(lang, ERRORS["lv"])
    if ("sign in to confirm" in m or "not a bot" in m or ("cookies" in m and "youtube" in m)
            or ("yt-dlp" in m and ("403" in m or "forbidden" in m or "unable to download video data" in m))):
        return e["yt_bot"]
    if "private video" in m or "video unavailable" in m or "age-restricted" in m or "age restricted" in m:
        return e["yt_private"]
    if "google drive" in m and ("quota" in m or "too many users" in m):
        return e["drive_quota"]
    if "google drive" in m or "gdown" in m:
        return e["drive_private"]
    if "no audio stream" in m:
        return e["no_audio"]
    if "provide a url" in m:
        return e["no_input"]
    if "request entity too large" in m or "413" in m:
        return e["too_big"]
    if "timed out" in m or "connection" in m or "temporary failure" in m or "name resolution" in m:
        return e["net"]
    if "ffmpeg" in m or "command failed" in m:
        return e["ffmpeg"]
    return e["unknown"].format(msg=(msg or "")[:300])


def stage_text(lang: str, stage: tuple | None, progress: float) -> str:
    if not stage:
        return t(lang, "st_start")
    key, params = stage
    params = dict(params or {})
    if key == "st_transcribe":
        params.setdefault("pct", int(progress * 100))
    return t(lang, key, **params)
