"""
Optional Google Cloud Storage persistence for Clipper.

Cloud Run's disk is ephemeral, so after every job `clips.zip` + `captions.md` (and `history.json`) are
copied to the bucket named by env GCS_BUCKET. When the variable is unset every function here is a no-op
and the app works purely on local disk.

Signed URLs: on Cloud Run the runtime service account has no private key, so URLs are signed through the
IAM `signBlob` API (requires roles/iam.serviceAccountTokenCreator on the runtime service account itself).
With a service-account JSON key (local dev) the key is used directly.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

BUCKET = os.environ.get("GCS_BUCKET", "").strip().removeprefix("gs://").strip("/")
PREFIX = os.environ.get("GCS_PREFIX", "clipper").strip("/")
HISTORY_BLOB = f"{PREFIX}/history.json" if PREFIX else "history.json"
SIGNED_URL_MINUTES = int(os.environ.get("GCS_SIGNED_URL_MINUTES", "120"))

_lock = threading.Lock()
_client = None
_creds = None
_sa_email: Optional[str] = None
_history_cache: tuple[float, list] | None = None


def enabled() -> bool:
    return bool(BUCKET)


def _get_client():
    global _client, _creds
    with _lock:
        if _client is None:
            import google.auth
            from google.cloud import storage

            _creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            _client = storage.Client(project=project, credentials=_creds)
        return _client


def _bucket():
    return _get_client().bucket(BUCKET)


def job_prefix(job_id: str) -> str:
    return f"{PREFIX}/{job_id}" if PREFIX else job_id


def blob_name(job_id: str, filename: str) -> str:
    return f"{job_prefix(job_id)}/{filename}"


def gs_uri(job_id: str, filename: str) -> str:
    return f"gs://{BUCKET}/{blob_name(job_id, filename)}"


# --------------------------------------------------------------------------- #
# upload / download
# --------------------------------------------------------------------------- #
def upload_file(local_path: str | Path, job_id: str, filename: Optional[str] = None,
                content_type: Optional[str] = None) -> Optional[str]:
    """Upload a local file into <prefix>/<job_id>/<filename>. Returns the gs:// URI or None."""
    if not enabled():
        return None
    local_path = Path(local_path)
    filename = filename or local_path.name
    blob = _bucket().blob(blob_name(job_id, filename))
    if content_type is None:
        content_type = {
            ".zip": "application/zip", ".md": "text/markdown; charset=utf-8", ".mp4": "video/mp4",
            ".json": "application/json",
        }.get(local_path.suffix.lower(), "application/octet-stream")
    blob.upload_from_filename(str(local_path), content_type=content_type, timeout=600)
    return gs_uri(job_id, filename)


def download_blob(name: str, dest: str | Path) -> Path:
    """Download an arbitrary blob (by full blob name) to `dest`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _bucket().blob(name).download_to_filename(str(dest), timeout=600)
    return dest


def download_gs_uri(uri: str, dest_dir: str | Path) -> Path:
    """Download gs://bucket/path (any bucket the service account can read) into dest_dir."""
    from google.cloud import storage

    assert uri.startswith("gs://")
    bucket_name, _, path = uri[5:].partition("/")
    if not path:
        raise ValueError("gs:// URI must point to an object, not a bucket")
    blob = storage.Blob.from_string(uri, client=_get_client())
    dest = Path(dest_dir) / Path(path).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest), timeout=1800)
    return dest


def exists(name: str) -> bool:
    if not enabled():
        return False
    try:
        return _bucket().blob(name).exists()
    except Exception:
        return False


def read_text(name: str) -> Optional[str]:
    if not enabled():
        return None
    try:
        return _bucket().blob(name).download_as_text(timeout=60)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# signed URLs
# --------------------------------------------------------------------------- #
def _service_account_email() -> Optional[str]:
    """Resolve the runtime service account e-mail (metadata server on Cloud Run)."""
    global _sa_email
    if _sa_email:
        return _sa_email
    email = getattr(_creds, "service_account_email", None)
    if email and email != "default":
        _sa_email = email
        return email
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            _sa_email = r.read().decode().strip()
            return _sa_email
    except Exception:
        return None


def signed_url(name: str, minutes: int = SIGNED_URL_MINUTES, download_name: Optional[str] = None) -> Optional[str]:
    """
    V4 signed GET URL for a blob (by full blob name). Returns None when signing is impossible;
    callers should then fall back to serving the file through the app.
    """
    if not enabled():
        return None
    try:
        from google.auth.transport.requests import Request

        blob = _bucket().blob(name)
        expiration = _dt.timedelta(minutes=minutes)
        disposition = f'attachment; filename="{download_name}"' if download_name else None
        creds = _creds
        # Credentials with a private key (service-account JSON) can sign directly
        if getattr(creds, "signer", None) is not None and getattr(creds, "signer_email", None):
            return blob.generate_signed_url(version="v4", expiration=expiration, method="GET",
                                            response_disposition=disposition)
        # Otherwise sign via IAM signBlob using the runtime service account's access token
        if creds is not None and (not creds.valid or creds.expired):
            creds.refresh(Request())
        email = _service_account_email()
        if not email or creds is None or not creds.token:
            return None
        return blob.generate_signed_url(version="v4", expiration=expiration, method="GET",
                                        service_account_email=email, access_token=creds.token,
                                        response_disposition=disposition)
    except Exception as e:  # pragma: no cover
        print(f"[gcs] signed URL failed for {name}: {e}", flush=True)
        return None


# --------------------------------------------------------------------------- #
# history.json in the bucket
# --------------------------------------------------------------------------- #
def load_history(max_age: float = 20.0) -> Optional[list]:
    """History list from the bucket (cached briefly), or None when GCS is disabled / unreadable."""
    global _history_cache
    if not enabled():
        return None
    now = time.time()
    if _history_cache and now - _history_cache[0] < max_age:
        return _history_cache[1]
    txt = read_text(HISTORY_BLOB)
    if txt is None:
        return [] if not exists(HISTORY_BLOB) else None
    try:
        data = json.loads(txt)
    except Exception:
        data = []
    _history_cache = (now, data)
    return data


def save_history(entries: list) -> bool:
    global _history_cache
    if not enabled():
        return False
    try:
        _bucket().blob(HISTORY_BLOB).upload_from_string(
            json.dumps(entries, ensure_ascii=False, indent=1), content_type="application/json", timeout=60)
        _history_cache = (time.time(), entries)
        return True
    except Exception as e:  # pragma: no cover
        print(f"[gcs] saving history failed: {e}", flush=True)
        return False


def delete_job(job_id: str):
    """Remove every object of a job (used when an entry falls off the 20-item history)."""
    if not enabled():
        return
    try:
        for blob in _get_client().list_blobs(BUCKET, prefix=job_prefix(job_id) + "/"):
            blob.delete()
    except Exception as e:  # pragma: no cover
        print(f"[gcs] delete {job_id} failed: {e}", flush=True)


# --------------------------------------------------------------------------- #
# logo library  (<prefix>/logos/<name>.png)
# --------------------------------------------------------------------------- #
def logos_prefix() -> str:
    return f"{PREFIX}/logos" if PREFIX else "logos"


def list_logos() -> list[str]:
    if not enabled():
        return []
    try:
        pre = logos_prefix() + "/"
        return [b.name[len(pre):] for b in _get_client().list_blobs(BUCKET, prefix=pre)
                if b.name != pre and b.name.lower().endswith((".png", ".webp"))]
    except Exception as e:  # pragma: no cover
        print(f"[gcs] listing logos failed: {e}", flush=True)
        return []


def upload_logo(local_path: str | Path, name: str) -> Optional[str]:
    if not enabled():
        return None
    blob = _bucket().blob(f"{logos_prefix()}/{name}")
    blob.upload_from_filename(str(local_path), content_type="image/png", timeout=120)
    return f"gs://{BUCKET}/{blob.name}"


def download_logo(name: str, dest: str | Path) -> bool:
    if not enabled():
        return False
    try:
        blob = _bucket().blob(f"{logos_prefix()}/{name}")
        if not blob.exists():
            return False
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest), timeout=120)
        return True
    except Exception as e:  # pragma: no cover
        print(f"[gcs] downloading logo {name} failed: {e}", flush=True)
        return False
