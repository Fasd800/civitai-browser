# civitai-browser.py

import gradio as gr
import requests
import os
import json
import re
import threading
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import modules.scripts as scripts  # noqa: F401 (kept for SD WebUI extension conventions)
from modules import shared, script_callbacks

# =============================================================================
# CONSTANTS
# =============================================================================
CIVITAI_API = "https://civitai.com/api/v1"
DOWNLOAD_URL = "https://civitai.com/api/download/models"

EXTENSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(EXTENSION_DIR, "settings.json")

_DISCORD_INVITE = "https://discord.gg/bqP8qVp8XS"
MAX_TABS = 5

MODEL_DIRS = {
    "Checkpoint": "models/Stable-diffusion",
    "LORA": "models/Lora",
    "TextualInversion": "embeddings",
    "Controlnet": "models/ControlNet",
    "Hypernetwork": "models/hypernetworks",
    "VAE": "models/VAE",
    "Poses": "models/Poses",
    "Wildcards": "models/Wildcards",
    "Other": "models/other",
}
_MODEL_DIRS_NORM = {k.strip().lower(): v for k, v in MODEL_DIRS.items()}

TYPE_COLORS = {
    "LORA": "#7c3aed",
    "Checkpoint": "#1d4ed8",
    "TextualInversion": "#0f766e",
    "Controlnet": "#b45309",
    "Hypernetwork": "#be123c",
    "VAE": "#0e7490",
    "Poses": "#4d7c0f",
    "Wildcards": "#6d28d9",
    "Other": "#374151",
}

_LAST_REQ_TS = 0.0
_RATE_MIN_INTERVAL = 0.5
_TAG_CACHE = {}
_SESSION = requests.Session()
_RETRY = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504], allowed_methods=frozenset(["GET"]))
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))
_SESSION.mount("http://", HTTPAdapter(max_retries=_RETRY))
DOWNLOAD_CANCEL_REQUESTED = False

def _safe_get(url, headers=None, params=None, timeout=15, stream=False):
    global _LAST_REQ_TS
    now = time.time()
    wait = _RATE_MIN_INTERVAL - (now - _LAST_REQ_TS)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQ_TS = time.time()
    r = _SESSION.get(url, headers=headers or {}, params=params, timeout=timeout, stream=stream)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        try:
            delay = float(ra)
        except Exception:
            delay = 2.0
        time.sleep(min(delay, 5.0))
        r = _SESSION.get(url, headers=headers or {}, params=params, timeout=timeout, stream=stream)
    r.raise_for_status()
    return r

# =============================================================================
# SETTINGS
# =============================================================================
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"api_key": "", "favorite_creators": []}


def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[CivitaiBrowser] Error saving settings: {e}")
        return False


def get_favorite_creators():
    return load_settings().get("favorite_creators", [])


def creator_dropdown_choices():
    return ["— All —"] + get_favorite_creators()


def get_model_dir(model_type):
    base = getattr(shared, "data_path", ".")
    key = (model_type or "Other").strip().lower()
    rel = _MODEL_DIRS_NORM.get(key, _MODEL_DIRS_NORM.get("other", "models/other"))
    return os.path.join(base, rel)


# =============================================================================
# URL PARSING
# =============================================================================
def parse_civitai_url(url: str):
    url = url.strip()
    m = re.search(r"civitai\.com/models/(\d+)", url)
    if not m:
        return None, None
    model_id = m.group(1)
    v = re.search(r"[?&]modelVersionId=(\d+)", url)
    version_id = v.group(1) if v else None
    return model_id, version_id


def fetch_model_by_id(model_id: str, api_key: str):
    headers = {}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    try:
        r = _safe_get(f"{CIVITAI_API}/models/{model_id}", headers=headers, timeout=15)
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, str(e)


# =============================================================================
# CIVITAI API HELPERS
# =============================================================================
def resolve_tag(query, headers):
    q = (query or "").strip()
    if q in _TAG_CACHE:
        return _TAG_CACHE[q]
    try:
        r = _safe_get(f"{CIVITAI_API}/tags", headers=headers, params={"query": q, "limit": 5}, timeout=10)
        items = r.json().get("items", [])
        if items:
            name = max(items, key=lambda x: x.get("modelCount", 0)).get("name", q)
            _TAG_CACHE[q] = name
            return name
    except Exception:
        pass
    return q


def _get_headers(api_key):
    return {"Authorization": f"Bearer {api_key.strip()}"} if api_key.strip() else {}


def _fetch_url(url, headers):
    try:
        r = _safe_get(url, headers=headers, timeout=15)
        data = r.json()
        meta = data.get("metadata", {})
        return data.get("items", []), meta, meta.get("nextPage", "")
    except Exception as e:
        print(f"[CivitaiBrowser] _fetch_url error: {e}")
        return [], {}, ""


def _matches_query(model, q: str) -> bool:
    return (
        q in model.get("name", "").lower()
        or any(q in t.lower() for t in model.get("tags", []))
        or any(q in v.get("name", "").lower() for v in model.get("modelVersions", []))
    )


def build_search_url(query, model_type, sort, content_levels, api_key, creator_filter, period="AllTime", use_tag=False):
    include_nsfw = any(lvl in content_levels for lvl in ["PG-13", "R", "X", "XXX"])
    params = {
        "limit": 20,
        "sort": sort,
        "period": period,
        "nsfw": str(include_nsfw).lower(),
    }

    if model_type != "All":
        params["types"] = model_type

    if creator_filter and creator_filter != "— All —":
        params["username"] = creator_filter
        params["limit"] = 100
    elif query.strip():
        if use_tag:
            params["tag"] = query.strip()
        else:
            params["query"] = query.strip()

    req = requests.Request("GET", f"{CIVITAI_API}/models", params=params)
    prepared = req.prepare()
    return prepared.url


def search_first_page(query, model_type, sort, content_levels, api_key, creator_filter, period="AllTime"):
    headers = _get_headers(api_key)
    creator_active = creator_filter and creator_filter != "— All —"

    if creator_active:
        url = build_search_url(query, model_type, sort, content_levels, api_key, creator_filter, period)
        items, meta, next_page = _fetch_url(url, headers)
        return items, meta, next_page, url

    if query.strip():
        url_query = build_search_url(query, model_type, sort, content_levels, api_key, creator_filter, period, use_tag=False)
        items_query, meta1, next_q = _fetch_url(url_query, headers)

        resolved = resolve_tag(query.strip(), headers)
        url_tag = build_search_url(resolved, model_type, sort, content_levels, api_key, creator_filter, period, use_tag=True)
        items_tag, meta2, next_t = _fetch_url(url_tag, headers)

        if len(items_query) < 5 and items_tag:
            return items_tag, meta2, next_t, url_tag

        if not items_query and not items_tag:
            return [], {}, "", url_query

        seen = set()
        items = []
        for item in (items_query + items_tag):
            mid = item.get("id")
            if mid is None or mid in seen:
                continue
            seen.add(mid)
            items.append(item)

        total_q = int(meta1.get("totalItems") or 0)
        total_t = int(meta2.get("totalItems") or 0)
        if total_q >= total_t:
            return items, meta1, next_q, url_query
        else:
            return items, meta2, next_t, url_tag

    url = build_search_url("", model_type, sort, content_levels, api_key, creator_filter, period)
    items, meta, next_page = _fetch_url(url, headers)
    return items, meta, next_page, url


def search_creator_on_civitai(query, api_key):
    headers = _get_headers(api_key)
    try:
        r = _safe_get(f"{CIVITAI_API}/creators", headers=headers, params={"query": query, "limit": 10}, timeout=10)
        return [i.get("username", "") for i in r.json().get("items", []) if i.get("username")]
    except Exception:
        return []


# =============================================================================
# MODEL HELPERS
# =============================================================================
def _version_label(v):
    return f"{v.get('name', '?')} — base: {v.get('baseModel', '?')}"


def get_version_by_choice(model, version_choice):
    versions = model.get("modelVersions", [])
    if not versions:
        return None
    for v in versions:
        if _version_label(v) == version_choice:
            return v
    return versions[0]


def get_trigger_words_for_version(version):
    return version.get("trainedWords", []) if version else []


def _has_thumbnail(model):
    skip_types = {"video"}
    skip_ext = {".mp4", ".webm", ".gif", ".mov", ".avi"}
    for ver in model.get("modelVersions", [])[:1]:
        for img in ver.get("images", []):
            if img.get("type", "image").lower() in skip_types:
                continue
            url = img.get("url", "")
            ext = os.path.splitext(url.split("?")[0])[1].lower()
            if ext in skip_ext:
                continue
            return True
    return False


def build_gallery_data(items):
    gallery = []
    for m in items:
        versions = m.get("modelVersions", []) or []
        sel_id = m.get("_civitai_selected_version_id", None)
        chosen = None
        if sel_id is not None:
            for v in versions:
                if str(v.get("id")) == str(sel_id):
                    chosen = v
                    break
        if chosen is None and versions:
            chosen = versions[0]

        thumb = _pick_version_preview_image_url(chosen or {})
        if thumb:
            gallery.append((thumb, m.get("name", "?")))
    return gallery


def _pick_version_preview_image_url(version: dict):
    if not version:
        return ""
    skip_types = {"video"}
    skip_ext = {".mp4", ".webm", ".gif", ".mov", ".avi"}
    for img in version.get("images", []) or []:
        if img.get("type", "image").lower() in skip_types:
            continue
        url = (img.get("url", "") or "").strip()
        if not url:
            continue
        ext = os.path.splitext(url.split("?")[0])[1].lower()
        if ext in skip_ext:
            continue
        return url
    return ""


# =============================================================================
# HTML BUILDERS
# =============================================================================
def build_trigger_words_html(words):
    if not words:
        return (
            "<div style='padding:8px 10px;background:#111;border-radius:8px;"
            "border:1px solid #1f2937;color:#6b7280;font-size:12px;font-style:italic'>"
            "No trigger words</div>"
        )

    pills = []
    for w in words:
        esc = (w or "").replace("'", "\\'").replace('"', "&quot;")
        pills.append(
            "<span "
            "data-word=\"" + esc + "\" "
            "onclick=\"(function(el){"
            "var txt=el.getAttribute('data-word')||'';"
            "if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(txt).then(function(){"
            "el.style.background='#166534';el.style.borderColor='#4ade80';setTimeout(function(){el.style.background='#1a2e1a';el.style.borderColor='#7c3aed';},600);"
            "}).catch(function(){"
            "var ta=document.createElement('textarea');ta.value=txt;ta.style.position='fixed';ta.style.left='-1000px';document.body.appendChild(ta);ta.focus();ta.select();try{document.execCommand('copy');}catch(e){}document.body.removeChild(ta);"
            "el.style.background='#166534';el.style.borderColor='#4ade80';setTimeout(function(){el.style.background='#1a2e1a';el.style.borderColor='#7c3aed';},600);"
            "});}else{"
            "var ta=document.createElement('textarea');ta.value=txt;ta.style.position='fixed';ta.style.left='-1000px';document.body.appendChild(ta);ta.focus();ta.select();try{document.execCommand('copy');}catch(e){}document.body.removeChild(ta);"
            "el.style.background='#166534';el.style.borderColor='#4ade80';setTimeout(function(){el.style.background='#1a2e1a';el.style.borderColor='#7c3aed';},600);"
            "}"
            "})(this)\" "
            "title='Click to copy' "
            "style='display:inline-block;margin:3px 4px 3px 0;padding:4px 10px;"
            "background:#1a2e1a;border:1px solid #7c3aed;border-radius:20px;"
            "color:#fbbf24;font-size:12px;font-family:monospace;cursor:pointer;"
            "user-select:none;transition:all 0.2s ease' "
            "onmouseover=\"this.style.background='#2d1f5e';this.style.borderColor='#a78bfa'\" "
            "onmouseout=\"this.style.background='#1a2e1a';this.style.borderColor='#7c3aed'\""
            f">{w}</span>"
        )

    return (
        "<div style='padding:8px 10px;background:#111;border-radius:8px;border:1px solid #1f2937;min-height:36px'>"
        "<div style='font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px'>"
        "Trigger words (click to copy)</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:2px'>"
        + "".join(pills)
        + "</div></div>"
    )


def sanitize_description_html(raw: str) -> str:
    if not raw:
        return ""
    safe = re.sub(r"<(script|style|iframe|object|embed|form|input|button)[^>]*?>.*?</\\1>", "", raw, flags=re.IGNORECASE | re.DOTALL)
    safe = re.sub(r"<(script|style|iframe|object|embed|form|input|button)[^>]*?/>", "", safe, flags=re.IGNORECASE)
    safe = re.sub(r"on\\w+\\s*=\\s*\"[^\"]*\"", "", safe, flags=re.IGNORECASE)
    safe = re.sub(r"on\\w+\\s*=\\s*'[^']*'", "", safe, flags=re.IGNORECASE)
    safe = re.sub(r"on\\w+\\s*=\\s*[^\\s>]+", "", safe, flags=re.IGNORECASE)
    return safe.strip()


def _has_meaningful_html(html: str) -> bool:
    if not html:
        return False
    txt = re.sub(r"<[^>]+>", "", html)
    txt = txt.replace("&nbsp;", " ").replace("\u00a0", " ")
    return bool(txt.strip())


def build_open_link_html(model):
    mid = model.get("id", "")
    if not mid:
        return ""
    url = f"https://civitai.com/models/{mid}"
    return (
        "<div style='margin:6px 0 10px'>"
        f"<a href='{url}' target='_blank' "
        "style='display:inline-block;padding:4px 10px;background:#1e2d3d;border:1px solid #1d4ed8;"
        "border-radius:12px;color:#60a5fa;font-size:12px;text-decoration:none;font-weight:600'>"
        "Open on CivitAI</a>"
        "</div>"
    )


def get_model_detail_html(model, version=None):
    if not model:
        return EMPTY_DETAIL

    if version is None and model.get("modelVersions"):
        version = model["modelVersions"][0]

    vername = version.get("name", "?") if version else "?"
    verbasemodel = version.get("baseModel", "?") if version else "?"
    verdate = (version.get("createdAt") or "")[:10] if version else ""

    rawdesc = model.get("description") or ""
    if not rawdesc and version:
        rawdesc = version.get("description") or ""
    safedesc = sanitize_description_html(rawdesc)

    desc_html = (
        "<details style='margin-top:10px'>"
        "<summary style='cursor:pointer;padding:8px 12px;background:#1e2a1e;border-radius:6px;"
        "border-left:3px solid #4ade80;color:#4ade80;font-size:12px;font-weight:700;"
        "list-style:none;user-select:none'>Description</summary>"
        "<div style='padding:10px 12px;background:#161f16;border-radius:0 0 6px 6px;"
        "color:#d1d5db;font-size:12px;line-height:1.8;border:1px solid #2a3a2a;border-top:none;"
        "max-height:340px;overflow-y:auto;word-break:break-word'>"
        "<style scoped>"
        ".civitai-desc h1,.civitai-desc h2,.civitai-desc h3{color:#e0e7ff;margin:10px 0 4px;font-size:13px;font-weight:700}"
        ".civitai-desc p{margin:4px 0}"
        ".civitai-desc ul,.civitai-desc ol{padding-left:18px;margin:4px 0}"
        ".civitai-desc li{margin:2px 0}"
        ".civitai-desc a{color:#60a5fa;text-decoration:underline}"
        ".civitai-desc strong,.civitai-desc b{color:#fff}"
        ".civitai-desc em,.civitai-desc i{color:#d1d5db}"
        ".civitai-desc code{background:#0d1117;padding:1px 5px;border-radius:4px;font-family:monospace;color:#a78bfa}"
        ".civitai-desc hr{border-color:#1f2937;margin:8px 0}"
        ".civitai-desc img{max-width:100%;border-radius:6px;margin:4px 0}"
        "</style>"
        f"<div class='civitai-desc'>{safedesc or '<i style=\"color:#6b7280\">No description available.</i>'}</div>"
        "</div></details>"
    )
    about_html = ""
    ver_desc = sanitize_description_html((version or {}).get("description") or "")
    if _has_meaningful_html(ver_desc):
        about_html = (
            "<details style='margin-top:10px'>"
            "<summary style='cursor:pointer;padding:8px 12px;background:#1b2332;border-radius:6px;"
            "border-left:3px solid #60a5fa;color:#60a5fa;font-size:12px;font-weight:700;"
            "list-style:none;user-select:none'>About this version</summary>"
            "<div style='padding:10px 12px;background:#121926;border-radius:0 0 6px 6px;"
            "color:#d1d5db;font-size:12px;line-height:1.8;border:1px solid #233046;border-top:none;"
            "max-height:260px;overflow-y:auto;word-break:break-word'>"
            f"<div class='civitai-desc'>{ver_desc}</div>"
            "</div></details>"
        )

    notes_html = ""
    ver_notes_raw = ""
    if version:
        for k in ["changelog", "changeNotes", "versionNotes", "notes", "changes", "about", "updateNotes"]:
            v = version.get(k)
            if isinstance(v, str) and v.strip():
                ver_notes_raw = v
                break
    ver_notes = sanitize_description_html(ver_notes_raw)
    if _has_meaningful_html(ver_notes):
        notes_html = (
            "<details style='margin-top:10px'>"
            "<summary style='cursor:pointer;padding:8px 12px;background:#2a2209;border-radius:6px;"
            "border-left:3px solid #fbbf24;color:#fbbf24;font-size:12px;font-weight:700;"
            "list-style:none;user-select:none'>Version changes or notes</summary>"
            "<div style='padding:10px 12px;background:#1a1407;border-radius:0 0 6px 6px;"
            "color:#d1d5db;font-size:12px;line-height:1.8;border:1px solid #3a2b10;border-top:none;"
            "max-height:260px;overflow-y:auto;word-break:break-word'>"
            f"<div class='civitai-desc'>{ver_notes}</div>"
            "</div></details>"
        )

    stats = model.get("stats", {}) or {}
    downloads = stats.get("downloadCount", 0)
    rating = float(stats.get("rating", 0) or 0)
    ratingcnt = int(stats.get("ratingCount", 0) or 0)
    creator = (model.get("creator") or {}).get("username", "NA")
    modeltype = model.get("type", "Other")
    typecolor = TYPE_COLORS.get(modeltype, "#374151")
    tags = model.get("tags", [])[:6]

    tags_html = ""
    if tags:
        tags_html = "<div style='margin-top:8px'>" + "".join(
            f"<span style='display:inline-block;padding:2px 8px;margin:2px;background:#1c1c2e;border:1px solid #374151;"
            f"border-radius:12px;color:#9ca3af;font-size:10px'>{t}</span>"
            for t in tags
        ) + "</div>"

    stars = ""
    if ratingcnt > 0:
        stars = (
            "<span style='background:#2a2209;border:1px solid #92400e;color:#fbbf24;"
            "padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600'>"
            f"{rating:.1f} ★ ({ratingcnt:,})</span>"
        )

    return (
        "<div style='padding:12px 14px;font-family:sans-serif;color:#e0e0e0'>"
        "<div style='margin-bottom:10px'>"
        "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px'>"
        f"<h3 style='margin:0;color:#fff;font-size:16px;line-height:1.3;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{model.get('name','NA')}</h3>"
        f"<span style='background:{typecolor};color:#fff;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:700;white-space:nowrap;flex-shrink:0'>{modeltype}</span>"
        "</div>"
        "<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap'>"
        f"<span style='background:#1e2d3d;border:1px solid #1d4ed8;color:#60a5fa;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600'>{creator}</span>"
        f"<span style='background:#1a2e1a;border:1px solid #166534;color:#34d399;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600'>{downloads:,} downloads</span>"
        f"{stars}"
        "</div>"
        f"{tags_html}"
        f"{about_html}"
        f"{notes_html}"
        f"{desc_html}"
        "</div></div>"
    )


def discord_banner_html():
    return (
        "<div style='padding:10px 14px;margin-top:10px;background:linear-gradient(135deg,#1e1f2e 0%,#2b2d42 100%);"
        "border:1px solid #5865F2;border-radius:10px;display:flex;align-items:center;gap:10px;"
        "box-shadow:0 2px 8px rgba(88,101,242,0.18)'>"
        "<div style='font-size:22px;flex-shrink:0'>💬</div>"
        "<div style='flex:1;min-width:0'>"
        "<div style='color:#fff;font-size:12px;font-weight:700;margin-bottom:2px'>Join the Discord server!</div>"
        "<div style='color:#a5b4fc;font-size:11px;line-height:1.4'>Request commissions and interact with community members.</div>"
        "</div>"
        f"<a href='{_DISCORD_INVITE}' target='_blank' style='flex-shrink:0;display:inline-block;padding:6px 14px;background:#5865F2;color:#fff;"
        "font-size:12px;font-weight:700;border-radius:8px;text-decoration:none;white-space:nowrap'"
        " onmouseover=\"this.style.background='#4752C4'\" onmouseout=\"this.style.background='#5865F2'\">Join Discord</a>"
        "</div>"
    )


def render_tab_bar(count, active):
    tabs_html = ""
    for i in range(count):
        is_active = i == active
        tab_class = "civitai-tab active" if is_active else "civitai-tab"

        close_btn = ""
        if count > 1:
            close_btn = (
                "<span "
                "class='civitai-tab-close' "
                "title='Close tab' "
                f"onclick=\"event.stopPropagation();var el=document.getElementById('civitai-close-btn-{i}');if(el) el.click();\""
                "aria-label='Close tab'"
                "><span class='civitai-tab-close-icon'>×</span></span>"
            )

        tabs_html += (
            f"<div class='{tab_class}' "
            f"data-tab-index='{i}' "
            f"title='Search {i+1}' "
            f"onclick=\"var el=document.getElementById('civitai-switch-btn-{i}');if(el) el.click();\" "
            f"onauxclick=\"if(event.button===1){{event.preventDefault();var el=document.getElementById('civitai-close-btn-{i}');if(el) el.click();}}\""
            f"><span class='civitai-tab-icon' aria-hidden='true'></span><span class='civitai-tab-title'>Search {i+1}</span>{close_btn}</div>"
        )

    if count < MAX_TABS:
        tabs_html += (
            "<div class='civitai-tab-add' "
            "title='New tab' "
            "onclick=\"var el=document.getElementById('civitai-add-btn');if(el) el.click();\" "
            "aria-label='New tab'"
            "><span class='civitai-tab-add-icon'>+</span></div>"
        )

    return f"<div class='civitai-tabstrip'>{tabs_html}</div>"


EMPTY_DETAIL = (
    "<div style='padding:60px 20px;color:#374151;text-align:center;font-size:13px;background:#0d1117;"
    "border-radius:10px;border:1px dashed #1f2937'>"
    "<div style='font-size:32px;margin-bottom:10px'>✨</div>"
    "Select a model from the gallery or paste a CivitAI URL above"
    "</div>"
)

# =============================================================================
# DOWNLOAD
# =============================================================================
def _pick_download_url_and_name(version: dict):
    files = version.get("files", []) if version else []
    if not files:
        return None, None
    primary = next((f for f in files if f.get("primary")), files[0])
    dl = primary.get("downloadUrl")
    name = primary.get("name")
    return dl, name


def _pick_first_image_url(version: dict):
    if not version:
        return None
    allowed = {".png", ".jpg", ".jpeg"}
    for img in version.get("images", []) or []:
        url = img.get("url") or ""
        if not url:
            continue
        ext = os.path.splitext(url.split("?")[0])[1].lower()
        if ext in allowed:
            return url
    return None


def _render_progress_html(percent, done, total, filename):
    percent = max(0, min(100, int(percent or 0)))
    done_mb = (done or 0) / 1024 / 1024
    total_mb = (total or 0) / 1024 / 1024
    label = f"{filename} — {done_mb:.1f} MB" + (f" / {total_mb:.1f} MB" if total else "")
    return (
        "<div style='margin-top:8px'>"
        "<div style='height:16px;background:#0f172a;border:1px solid #1f2937;border-radius:10px;overflow:hidden'>"
        f"<div style='height:100%;width:{percent}%;background:#3b82f6;transition:width 0.2s ease'></div>"
        "</div>"
        f"<div style='font-size:11px;color:#9ca3af;margin-top:4px'>{label}</div>"
        "</div>"
    )

def download_model(search_data, version_choice, api_key):
    global DOWNLOAD_CANCEL_REQUESTED
    items = search_data.get("items", [])
    idx = search_data.get("selected_index", 0)
    if not items or idx >= len(items):
        yield "", "No model selected."
        return

    model = items[idx]
    version = get_version_by_choice(model, version_choice)
    if not version:
        yield "", "No version found."
        return

    model_type = model.get("type", "Other")
    save_dir = get_model_dir(model_type)
    os.makedirs(save_dir, exist_ok=True)

    ver_id = version.get("id")
    dl_url, filename = _pick_download_url_and_name(version)
    if not filename:
        filename = f"{model.get('id','model')}_{ver_id or 'latest'}.safetensors"

    dest = os.path.join(save_dir, filename)
    if os.path.exists(dest):
        existing = os.path.getsize(dest)
        yield _render_progress_html(100, existing, existing, filename), f"Already exists: {filename}"
        return

    if not dl_url and ver_id:
        dl_url = f"{DOWNLOAD_URL}/{ver_id}"
    if not dl_url:
        yield "", "No download URL found for this version."
        return

    headers = _get_headers(api_key)

    try:
        DOWNLOAD_CANCEL_REQUESTED = False
        with _safe_get(dl_url, headers=headers, stream=True, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            yield _render_progress_html(0, 0, total, filename), f"Starting download: {filename}"
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    pct = (done / total) * 100.0 if total > 0 else 0
                    yield _render_progress_html(pct, done, total, filename), f"Downloading: {filename} ({done/1024/1024:.1f} MB)"
                    if DOWNLOAD_CANCEL_REQUESTED:
                        try:
                            f.close()
                        except Exception:
                            pass
                        try:
                            if os.path.exists(dest):
                                os.remove(dest)
                        except Exception:
                            pass
                        yield "", "Download cancelled."
                        return

        size_mb = done / 1024 / 1024 if done else 0
        total_mb = total / 1024 / 1024 if total else 0
        msg = (f"Downloaded: {filename} ({size_mb:.1f}/{total_mb:.1f} MB) to {save_dir}" if total_mb > 0 else f"Downloaded: {filename} ({size_mb:.1f} MB) to {save_dir}")

        if (model_type or "").strip().lower() == "lora":
            img_url = _pick_first_image_url(version)
            if img_url:
                try:
                    img_ext = os.path.splitext(img_url.split("?")[0])[1].lower() or ".jpg"
                    if img_ext not in {".png", ".jpg", ".jpeg"}:
                        img_ext = ".jpg"
                    img_name = f"{os.path.splitext(filename)[0]}{img_ext}"
                    img_dest = os.path.join(save_dir, img_name)
                    if not os.path.exists(img_dest):
                        with _safe_get(img_url, headers=headers, stream=True, timeout=30) as ir:
                            with open(img_dest, "wb") as outf:
                                for chunk in ir.iter_content(chunk_size=1 << 20):
                                    if chunk:
                                        outf.write(chunk)
                        msg += f"\nPreview saved: {img_name}"
                    else:
                        msg += f"\nPreview exists: {img_name}"
                except Exception as ie:
                    msg += f"\nPreview download failed: {ie}"

        yield _render_progress_html(100, done, total, filename), msg
        return

    except Exception as e:
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except Exception:
            pass
        yield "", f"Download failed: {e}"
        return


def stop_download():
    global DOWNLOAD_CANCEL_REQUESTED
    DOWNLOAD_CANCEL_REQUESTED = True
    return "", "Stopping current download..."


# =============================================================================
# UI PANELS
# =============================================================================
def make_panel_components(i, api_key_state):
    with gr.Column(visible=i == 0, elem_id=f"civitai-panel-{i}") as col:
        with gr.Group():
            gr.Markdown("Filters or Load by URL")
            with gr.Row():
                url_input = gr.Textbox(
                    label="Model URL",
                    show_label=False,
                    placeholder="Paste a CivitAI model or version URL",
                    elem_id=f"civitai-url-input-{i}",
                    scale=5,
                )
                url_btn = gr.Button("🔗 Load from URL", variant="secondary", scale=1, min_width=120, elem_id=f"civitai-url-btn-{i}")

            url_status = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                lines=1,
                visible=False,
                placeholder="URL status",
            )

            with gr.Row():
                model_type = gr.Dropdown(
                    label="Type",
                    choices=["All", "Checkpoint", "LORA", "TextualInversion", "Controlnet", "Hypernetwork", "VAE", "Poses", "Wildcards", "Other"],
                    value="All",
                    scale=2,
                )
                sort = gr.Dropdown(
                    label="Sort by",
                    choices=["Most Downloaded", "Highest Rated", "Newest", "Most Liked", "Most Discussed"],
                    value="Newest",
                    scale=2,
                )
                period = gr.Dropdown(
                    label="Period",
                    choices=["AllTime", "Year", "Month", "Week", "Day"],
                    value="AllTime",
                    scale=2,
                )
                creator_filter = gr.Dropdown(
                    label="Creator",
                    choices=creator_dropdown_choices(),
                    value="— All —",
                    scale=2,
                )
                content_levels = gr.CheckboxGroup(
                    label="",
                    show_label=False,
                    choices=["PG", "PG-13", "R", "X", "XXX"],
                    value=["PG", "PG-13", "R", "X", "XXX"],
                    scale=3,
                    elem_classes=["content-center"],
                )
                search_btn = gr.Button("🔍 Load models", variant="primary", scale=4, min_width=220, elem_classes=["btn-load"])

        with gr.Group():
            gr.Markdown("Load Models and Keyword Filter")
            with gr.Row():
                query = gr.Textbox(
                    label="Keyword filter",
                    show_label=False,
                    placeholder="Filter by keyword (character name, series name...)",
                    elem_id=f"civitai-query-{i}",
                    scale=5,
                )
                refine_btn = gr.Button("✨ Refine results", variant="secondary", scale=1, min_width=110, elem_classes=["btn-refine-orange"])

            # removed load status bar

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("Results")
                gallery = gr.Gallery(
                    label="",
                    show_label=False,
                    columns=2,
                    rows=4,
                    height=480,
                    elem_id=f"civitai-gallery-{i}",
                    object_fit="cover",
                    interactive=False,
                    allow_preview=True,
                )

                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous", variant="secondary", scale=1, min_width=90)
                    page_info = gr.Textbox(label="", show_label=False, interactive=False, lines=1, value="", scale=3, placeholder="Page info")
                    next_btn = gr.Button("➡️ Next", variant="secondary", scale=1, min_width=90)

                gr.HTML(discord_banner_html())

            with gr.Column(scale=3):
                gr.Markdown("Model details")
                version_selector = gr.Dropdown(
                    label="Version",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False,
                )
                model_info = gr.HTML(EMPTY_DETAIL)
                trigger_html = gr.HTML(build_trigger_words_html([]))
                open_link_html = gr.HTML("")
                selected_url = gr.Textbox(value="", visible=False, elem_id=f"civitai-selected-url-{i}")

                gr.HTML('<hr style="border-color:#1f2937;margin:0">')

                with gr.Row():
                    download_btn = gr.Button("⬇️ Download model", variant="primary", scale=3, min_width=220, elem_classes=["btn-download"])
                    stop_btn = gr.Button("⏹️ Stop download", variant="secondary", scale=1, min_width=140)
                    send_tab_btn = gr.Button("📤 Send to new tab", variant="secondary", scale=1, min_width=170, elem_id=f"civitai-send-tab-{i}")
                dl_progress_html = gr.HTML("")

                dl_status = gr.Textbox(
                    label="Download status",
                    show_label=True,
                    interactive=False,
                    lines=3,
                    placeholder="Download status appears here.",
                )

        # State
        search_data = gr.State(
            {
                "items": [],
                "metadata": {},
                "all_items": [],
                "next_page": "",
                "first_page": "",
                "query": "",
                "selected_index": 0,
            }
        )

        # Events
        def on_gallery_select(evt: gr.SelectData, sd):
            items = sd.get("items", [])
            if not items or evt.index is None or evt.index >= len(items):
                return (
                    EMPTY_DETAIL,
                    build_trigger_words_html([]),
                    "",
                    "",
                    gr.update(visible=False, interactive=False, choices=[], value=None),
                    sd,
                )

            model = items[evt.index]
            versions = model.get("modelVersions", []) or []
            choices = [_version_label(v) for v in versions]
            sel_version = versions[0] if versions else None
            val = choices[0] if choices else None
            mid = model.get("id", "")
            vid = (sel_version or {}).get("id")
            sel_url = (f"https://civitai.com/models/{mid}" if mid else "") + (f"?modelVersionId={vid}" if mid and vid else "")

            sd2 = dict(sd)
            sd2["selected_index"] = int(evt.index)

            return (
                get_model_detail_html(model, sel_version),
                build_trigger_words_html(get_trigger_words_for_version(sel_version)),
                build_open_link_html(model),
                sel_url,
                gr.update(choices=choices, value=val, visible=True, interactive=len(choices) > 1),
                sd2,
            )

        gallery.select(
            fn=on_gallery_select,
            inputs=[search_data],
            outputs=[model_info, trigger_html, open_link_html, selected_url, version_selector, search_data],
        )

        def on_version_change(vc, sd):
            items = sd.get("items", [])
            idx = sd.get("selected_index", 0)
            if not items or idx >= len(items):
                return [], EMPTY_DETAIL, build_trigger_words_html([]), "", sd

            model = items[idx]
            v = get_version_by_choice(model, vc)
            mid = model.get("id", "")
            vid = (v or {}).get("id")
            sel_url = (f"https://civitai.com/models/{mid}" if mid else "") + (f"?modelVersionId={vid}" if mid and vid else "")
            m2 = dict(model)
            m2["_civitai_selected_version_id"] = vid
            items2 = list(items)
            items2[idx] = m2
            sd2 = dict(sd)
            sd2["items"] = items2

            return (
                build_gallery_data(items2),
                get_model_detail_html(m2, v),
                build_trigger_words_html(get_trigger_words_for_version(v)),
                sel_url,
                sd2,
            )

        version_selector.change(
            fn=on_version_change,
            inputs=[version_selector, search_data],
            outputs=[gallery, model_info, trigger_html, selected_url, search_data],
        )

        def load_from_url(url, api_key):
            empty_sd = {
                "items": [],
                "metadata": {},
                "all_items": [],
                "next_page": "",
                "first_page": "",
                "query": "",
                "selected_index": 0,
            }

            model_id, version_id = parse_civitai_url(url)
            if not model_id:
                return [], gr.update(value="URL not recognized.", visible=True), gr.update(visible=False, interactive=False), EMPTY_DETAIL, build_trigger_words_html([]), "", "", empty_sd

            model, err = fetch_model_by_id(model_id, api_key)
            if err or not model:
                return [], gr.update(value=(err or "Not found."), visible=True), gr.update(visible=False, interactive=False), EMPTY_DETAIL, build_trigger_words_html([]), "", "", empty_sd

            versions = model.get("modelVersions", []) or []
            ver_choices = [_version_label(v) for v in versions]

            selected_ver = None
            if version_id:
                for v in versions:
                    if str(v.get("id")) == str(version_id):
                        selected_ver = v
                        break
            if selected_ver is None and versions:
                selected_ver = versions[0]

            ver_val = _version_label(selected_ver) if selected_ver else (ver_choices[0] if ver_choices else None)
            mid = model.get("id", "")
            vid = (selected_ver or {}).get("id")
            sel_url = (f"https://civitai.com/models/{mid}" if mid else "") + (f"?modelVersionId={vid}" if mid and vid else "")
            m2 = dict(model)
            m2["_civitai_selected_version_id"] = vid

            new_sd = {
                "items": [m2],
                "metadata": {"totalItems": 1},
                "all_items": [m2],
                "next_page": "",
                "first_page": "",
                "query": "",
                "selected_index": 0,
            }

            return (
                build_gallery_data([m2]),
                gr.update(value=f"Loaded: {model.get('name','?')}", visible=True),
                gr.update(choices=ver_choices, value=ver_val, visible=True, interactive=len(ver_choices) > 1),
                get_model_detail_html(m2, selected_ver),
                build_trigger_words_html(get_trigger_words_for_version(selected_ver)),
                build_open_link_html(m2),
                sel_url,
                new_sd,
            )

        url_btn.click(
            fn=load_from_url,
            inputs=[url_input, api_key_state],
            outputs=[gallery, url_status, version_selector, model_info, trigger_html, open_link_html, selected_url, search_data],
        )
        url_input.submit(
            fn=load_from_url,
            inputs=[url_input, api_key_state],
            outputs=[gallery, url_status, version_selector, model_info, trigger_html, open_link_html, selected_url, search_data],
        )

        def do_search(q, mt, srt, levels, api_key, creator, per, sd):
            items, meta, next_page, first_page = search_first_page(q, mt, srt, levels, api_key, creator, per)
            visible_items = [m for m in items if _has_thumbnail(m)]
            creator_active = creator and creator != "— All —"
            all_loaded = list(visible_items)
            if creator_active and next_page:
                headers = _get_headers(api_key)
                seen = {m.get("id") for m in all_loaded if m.get("id") is not None}
                pages = 1
                while next_page:
                    pages += 1
                    items2, meta2, next2 = _fetch_url(next_page, headers)
                    visible2 = [m for m in items2 if _has_thumbnail(m)]
                    for m in visible2:
                        mid = m.get("id")
                        if mid is None or mid in seen:
                            continue
                        seen.add(mid)
                        all_loaded.append(m)
                    meta = meta2 or meta
                    next_page = next2
                    if pages >= 50 or len(all_loaded) >= 5000:
                        break

            if creator_active:
                total = meta.get("totalItems", len(all_loaded))
                page_lbl = f"Loaded {len(all_loaded)} of {total} results" if all_loaded else "No results found."
            else:
                total = meta.get("totalItems", len(visible_items))
                page_lbl = f"Page 1: {len(visible_items)} of {total} results" if visible_items else "No results found."

            new_sd = dict(sd)
            new_sd.update(
                {
                    "items": (all_loaded if creator_active else visible_items),
                    "metadata": meta,
                    "all_items": (all_loaded if creator_active else visible_items),
                    "next_page": ("" if creator_active else next_page),
                    "first_page": first_page,
                    "query": q,
                    "selected_index": 0,
                }
            )

            return (
                build_gallery_data(all_loaded if creator_active else visible_items),
                gr.update(value=page_lbl, visible=True),
                EMPTY_DETAIL,
                build_trigger_words_html([]),
                gr.update(visible=False, choices=[], value=None),
                new_sd,
            )

        search_btn.click(
            fn=do_search,
            inputs=[query, model_type, sort, content_levels, api_key_state, creator_filter, period, search_data],
            outputs=[gallery, page_info, model_info, trigger_html, version_selector, search_data],
        )
        query.submit(
            fn=do_search,
            inputs=[query, model_type, sort, content_levels, api_key_state, creator_filter, period, search_data],
            outputs=[gallery, page_info, model_info, trigger_html, version_selector, search_data],
        )

        def do_next(sd, api_key):
            next_url = sd.get("next_page", "")
            if not next_url:
                return build_gallery_data(sd.get("items", [])), gr.update(value="No more pages.", visible=True), EMPTY_DETAIL, build_trigger_words_html([]), gr.update(visible=False, choices=[], value=None), sd

            headers = _get_headers(api_key)
            items, meta, next2 = _fetch_url(next_url, headers)
            visible_items = [m for m in items if _has_thumbnail(m)]
            all_items = (sd.get("all_items") or []) + visible_items
            total = meta.get("totalItems", 0)
            page_lbl = f"{len(all_items)} of {total} loaded" if total else f"{len(all_items)} loaded"

            new_sd = dict(sd)
            new_sd.update({"items": visible_items, "metadata": meta, "all_items": all_items, "next_page": next2, "selected_index": 0})
            return build_gallery_data(visible_items), gr.update(value=page_lbl, visible=True), EMPTY_DETAIL, build_trigger_words_html([]), gr.update(visible=False, choices=[], value=None), new_sd

        next_btn.click(
            fn=do_next,
            inputs=[search_data, api_key_state],
            outputs=[gallery, page_info, model_info, trigger_html, version_selector, search_data],
        )

        def do_prev(sd, api_key):
            first_url = sd.get("first_page", "")
            if not first_url:
                return build_gallery_data(sd.get("items", [])), gr.update(value="Already on first page.", visible=True), EMPTY_DETAIL, build_trigger_words_html([]), gr.update(visible=False, choices=[], value=None), sd

            headers = _get_headers(api_key)
            items, meta, next2 = _fetch_url(first_url, headers)
            visible_items = [m for m in items if _has_thumbnail(m)]
            total = meta.get("totalItems", len(visible_items))
            page_lbl = f"Page 1: {len(visible_items)} of {total} results" if visible_items else "No results."

            new_sd = dict(sd)
            new_sd.update({"items": visible_items, "metadata": meta, "all_items": visible_items, "next_page": next2, "selected_index": 0})
            return build_gallery_data(visible_items), gr.update(value=page_lbl, visible=True), EMPTY_DETAIL, build_trigger_words_html([]), gr.update(visible=False, choices=[], value=None), new_sd

        prev_btn.click(
            fn=do_prev,
            inputs=[search_data, api_key_state],
            outputs=[gallery, page_info, model_info, trigger_html, version_selector, search_data],
        )

        def do_refine(q, sd, api_key):
            all_items = sd.get("all_items", []) or []
            if not all_items:
                items, meta, next_page, first_page = search_first_page(q, "All", "Most Downloaded", ["Safe"], api_key, "— All —", "AllTime")
                visible_items = [m for m in items if _has_thumbnail(m)]
                all_items = visible_items
                sd = dict(sd)
                sd.update({"items": visible_items, "metadata": meta, "all_items": visible_items, "next_page": next_page, "first_page": first_page, "selected_index": 0})

            if not q.strip():
                matched = all_items
            else:
                qq = q.strip().lower()
                matched = [m for m in all_items if _matches_query(m, qq)]

            sd2 = dict(sd)
            sd2["items"] = matched
            sd2["selected_index"] = 0

            page_lbl = f"{len(matched)} matches from {len(all_items)} cached"

            return build_gallery_data(matched), gr.update(value=page_lbl, visible=True), EMPTY_DETAIL, build_trigger_words_html([]), gr.update(visible=False, choices=[], value=None), sd2

        refine_btn.click(
            fn=do_refine,
            inputs=[query, search_data, api_key_state],
            outputs=[gallery, page_info, model_info, trigger_html, version_selector, search_data],
        )

        def clear_tab():
            empty_sd = {
                "items": [],
                "metadata": {},
                "all_items": [],
                "next_page": "",
                "first_page": "",
                "query": "",
                "selected_index": 0,
            }
            return (
                "",
                "",
                gr.update(value="", visible=False),
                gr.update(choices=[], value=None, visible=False),
                EMPTY_DETAIL,
                build_trigger_words_html([]),
                "",
                empty_sd,
            )

        download_btn.click(
            fn=download_model,
            inputs=[search_data, version_selector, api_key_state],
            outputs=[dl_progress_html, dl_status],
        )
        stop_btn.click(
            fn=stop_download,
            inputs=[],
            outputs=[dl_progress_html, dl_status],
        )

    return col


# =============================================================================
# CSS
# =============================================================================
STYLE_PATH = os.path.join(EXTENSION_DIR, "style.css")
def _load_css():
    try:
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""
CSS = _load_css()


# =============================================================================
# MAIN UI TAB
# =============================================================================
def on_ui_tabs():
    settings = load_settings()

    with gr.Blocks(analytics_enabled=False, css=CSS) as civitai_tab:
        api_key_state = gr.State(settings.get("api_key", ""))

        tab_count = gr.State(1)
        active_tab = gr.State(0)

        with gr.Tabs():
            with gr.TabItem("Civitai Browser"):
                tab_bar = gr.HTML(render_tab_bar(1, 0))

                # Hidden-but-rendered controls (kept for JS-driven switching)
                with gr.Row(elem_classes=["civitai-hidden-controls"]):
                    add_btn = gr.Button("add", elem_id="civitai-add-btn", visible=False)
                    close_btns = [
                        gr.Button(f"close-{i}", elem_id=f"civitai-close-btn-{i}", visible=False) for i in range(MAX_TABS)
                    ]
                    switch_btns = [
                        gr.Button(f"switch-{i}", elem_id=f"civitai-switch-btn-{i}", visible=False) for i in range(MAX_TABS)
                    ]

                panels = [make_panel_components(i, api_key_state) for i in range(MAX_TABS)]
                panel_cols = list(panels)

                def _vis_updates(count, active):
                    return [gr.update(visible=(j < count and j == active)) for j in range(MAX_TABS)]

                def do_add(count, active):
                    if count >= MAX_TABS:
                        return [count, active, render_tab_bar(count, active)] + _vis_updates(count, active)
                    new_count = count + 1
                    new_active = count
                    return [new_count, new_active, render_tab_bar(new_count, new_active)] + _vis_updates(new_count, new_active)

                def do_close(idx, count, active):
                    if count <= 1:
                        return [count, active, render_tab_bar(count, active)] + _vis_updates(count, active)
                    new_count = count - 1
                    new_active = active
                    if active == idx:
                        new_active = min(idx, new_count - 1)
                    elif active > idx:
                        new_active = active - 1
                    return [new_count, new_active, render_tab_bar(new_count, new_active)] + _vis_updates(new_count, new_active)

                def do_switch(idx, count, active):
                    if idx >= count:
                        return [count, active, render_tab_bar(count, active)] + _vis_updates(count, active)
                    return [count, idx, render_tab_bar(count, idx)] + _vis_updates(count, idx)

                shared_outputs = [tab_count, active_tab, tab_bar] + panel_cols

                add_btn.click(
                    fn=do_add,
                    inputs=[tab_count, active_tab],
                    outputs=shared_outputs,
                )

                for i in range(MAX_TABS):
                    close_btns[i].click(
                        fn=lambda cnt, act, iii=i: do_close(iii, cnt, act),
                        inputs=[tab_count, active_tab],
                        outputs=shared_outputs,
                    )
                    switch_btns[i].click(
                        fn=lambda cnt, act, iii=i: do_switch(iii, cnt, act),
                        inputs=[tab_count, active_tab],
                        outputs=shared_outputs,
                    )

            with gr.TabItem("Settings"):
                gr.Markdown("API Key")
                api_key_input = gr.Textbox(
                    label="CivitAI API Key",
                    placeholder="Paste your API key here",
                    value=settings.get("api_key", ""),
                    type="password",
                )
                save_api_btn = gr.Button("💾 Save API Key", variant="primary")
                api_save_status = gr.Textbox(label="", show_label=False, interactive=False, lines=1, placeholder="Save status")

                def save_api_key(key):
                    s = load_settings()
                    s["api_key"] = (key or "").strip()
                    ok = save_settings(s)
                    return ("API key saved." if ok else "Failed to save."), s["api_key"]

                save_api_btn.click(fn=save_api_key, inputs=[api_key_input], outputs=[api_save_status, api_key_state])

                gr.Markdown("Favorite Creators")
                with gr.Row():
                    new_favorite_input = gr.Textbox(label="", show_label=False, placeholder="Enter creator username", scale=4)
                    add_creator_btn = gr.Button("➕ Add", variant="secondary", scale=1, min_width=80)
                favorites_list = gr.Dropdown(label="Favorite creators", choices=get_favorite_creators(), value=None, interactive=True)
                remove_creator_btn = gr.Button("🗑️ Remove selected", variant="secondary")
                creator_status = gr.Textbox(label="", show_label=False, interactive=False, lines=1)

                def add_creator(username):
                    if not username:
                        return gr.update(), "No creator entered."
                    s = load_settings()
                    favs = s.get("favorite_creators", [])
                    if username not in favs:
                        favs.append(username)
                    s["favorite_creators"] = favs
                    save_settings(s)
                    return gr.update(choices=favs, value=None), f"Added: {username}"

                add_creator_btn.click(
                    fn=add_creator,
                    inputs=[new_favorite_input],
                    outputs=[favorites_list, creator_status],
                )

                def remove_creator(username):
                    if not username:
                        return gr.update(), "No creator selected."
                    s = load_settings()
                    favs = [f for f in s.get("favorite_creators", []) if f != username]
                    s["favorite_creators"] = favs
                    save_settings(s)
                    return gr.update(choices=favs, value=None), f"Removed: {username}"

                remove_creator_btn.click(
                    fn=remove_creator,
                    inputs=[favorites_list],
                    outputs=[favorites_list, creator_status],
                )

    return [(civitai_tab, "Civitai Browser", "civitai_browser")]


script_callbacks.on_ui_tabs(on_ui_tabs)
