"""
Small YAML settings store (campaigns.yaml, settings.yaml, rules.yaml) mirrored to the GCS bucket when
one is configured, so user settings survive Cloud Run instance restarts.

Rules: local file is the working copy. `sync_down()` at startup pulls the bucket copies over the local
ones (if present); every `save()` writes locally and uploads.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import yaml

import gcs

BASE_DIR = Path(__file__).resolve().parent
CAMPAIGNS_PATH = BASE_DIR / "campaigns.yaml"
SETTINGS_PATH = BASE_DIR / "settings.yaml"
RULES_PATH = BASE_DIR / "rules.yaml"
LOGOS_DIR = BASE_DIR / "logos"

_lock = threading.Lock()

DEFAULT_SETTINGS: dict[str, Any] = {
    "language": "lv",            # lv | en
    "expert_mode": False,
}

DEFAULT_CAMPAIGN: dict[str, Any] = {
    "name": "",
    "campaign_logo": "",          # file name inside logos/ ("" = none)
    "hashtags": "#shorts #viral",
    "caption_template": "{hook}\n\n{title} #Ad\n\n{hashtags}\n",
    "clips_per_video": 4,
    "keywords_boost": [],
}


def _blob(name: str) -> str:
    return f"{gcs.PREFIX}/config/{name}" if gcs.PREFIX else f"config/{name}"


def read_yaml(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return default if data is None else data
    except Exception as e:  # noqa: BLE001
        print(f"[store] cannot read {path.name}: {e}", flush=True)
    return default


def write_yaml(path: Path, data: Any, text: Optional[str] = None):
    """Write locally (text verbatim if given, else dumped) and mirror to the bucket."""
    with _lock:
        body = text if text is not None else yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        path.write_text(body, encoding="utf-8")
        if gcs.enabled():
            try:
                gcs.write_text(_blob(path.name), body)
            except Exception as e:  # noqa: BLE001
                print(f"[store] GCS mirror of {path.name} failed: {e}", flush=True)


def sync_down(names: tuple[str, ...] = ("rules.yaml", "campaigns.yaml", "settings.yaml")):
    """Pull config files from the bucket over the local copies (startup)."""
    if not gcs.enabled():
        return
    for name in names:
        try:
            txt = gcs.read_text(_blob(name))
            if txt:
                (BASE_DIR / name).write_text(txt, encoding="utf-8")
                print(f"[store] pulled {name} from gs://{gcs.BUCKET}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[store] pull {name} failed: {e}", flush=True)


# ---- settings --------------------------------------------------------------
def load_settings() -> dict:
    data = read_yaml(SETTINGS_PATH, {})
    out = dict(DEFAULT_SETTINGS)
    out.update(data if isinstance(data, dict) else {})
    if out["language"] not in ("lv", "en"):
        out["language"] = "lv"
    return out


def save_settings(**changes):
    s = load_settings()
    s.update(changes)
    write_yaml(SETTINGS_PATH, s)
    return s


# ---- campaigns --------------------------------------------------------------
def load_campaigns() -> list[dict]:
    data = read_yaml(CAMPAIGNS_PATH, {})
    items = data.get("campaigns", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        if not isinstance(it, dict) or not it.get("name"):
            continue
        c = dict(DEFAULT_CAMPAIGN)
        c.update(it)
        out.append(c)
    return out


def save_campaigns(items: list[dict]):
    write_yaml(CAMPAIGNS_PATH, {"campaigns": items})


def get_campaign(name: str) -> Optional[dict]:
    return next((c for c in load_campaigns() if c["name"] == name), None)


def upsert_campaign(c: dict) -> list[dict]:
    c = {**DEFAULT_CAMPAIGN, **c}
    c["name"] = str(c["name"]).strip()
    if not c["name"]:
        raise ValueError("name required")
    items = [x for x in load_campaigns() if x["name"] != c["name"]]
    items.append(c)
    items.sort(key=lambda x: x["name"].lower())
    save_campaigns(items)
    return items


def delete_campaign(name: str) -> list[dict]:
    items = [x for x in load_campaigns() if x["name"] != name]
    save_campaigns(items)
    return items
