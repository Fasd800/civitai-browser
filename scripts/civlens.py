# CivLens

import gradio as gr
import requests
import os
import json
import re
import threading
import time
import html
from urllib.parse import urlparse
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
_DOWNLOAD_JOBS = {}
_DOWNLOAD_JOBS_LOCK = threading.Lock()

def _is_allowed_url(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    if not host or not host.endswith("civitai.com"):
        return False
    if parsed.username or parsed.password:
        return False
    return True

def _escape_html(text) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)

def _sanitize_filename(name: str) -> str:
    clean = os.path.basename(str(name or ""))
    clean = clean.replace("\x00", "")
    clean = re.sub(r"[<>:\"/\\\\|?*\n\r\t]+", "_", clean).strip()
    if not clean or clean in {".", ".."}:
        return "model.safetensors"
    return clean[:180]

def _safe_join(base: str, name: str) -> str:
    base_abs = os.path.abspath(base)
    dest = os.path.abspath(os.path.join(base_abs, name))
    if os.path.commonpath([base_abs, dest]) != base_abs:
        return os.path.join(base_abs, os.path.basename(name))
    return dest

def _safe_get(url, headers=None, params=None, timeout=15, stream=False):
    if not _is_allowed_url(url):
        raise ValueError("Blocked URL")
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
        print(f"[CivLens] Error saving settings: {e}")
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
        print(f"[CivLens] _fetch_url error: {e}")
        return [], {}, ""


def _matches_query(model, q: str) -> bool:
    return (
        q in model.get("name", "").lower()
        or any(q in t.lower() for t in model.get("tags", []))
        or any(q in v.get("name", "").lower() for v in model.get("modelVersions", []))
    )


def _parse_tag_list(s: str):
    raw = (s or "").strip()
    if not raw:
        return []
    parts = re.split(r"[,\n]+", raw)
    out = []
    for p in parts:
        t = (p or "").strip()
        if t:
            out.append(t)
    seen = set()
    dedup = []
    for t in out:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(t)
    return dedup


def _model_matches_tags(model, required_tags):
    if not required_tags:
        return True
    tags = [t.lower() for t in (model.get("tags") or [])]
    tagset = set(tags)
    for t in required_tags:
        if (t or "").lower() not in tagset:
            return False
    return True


def _model_matches_any_tag(model, any_tags):
    if not any_tags:
        return True
    tags = [t.lower() for t in (model.get("tags") or [])]
    tagset = set(tags)
    for t in any_tags:
        if (t or "").lower() in tagset:
            return True
    return False


def _model_matches_base_model(model, base_model_value: str):
    bm = (base_model_value or "").strip()
    if not bm or bm == "Any":
        return True
    want = bm.lower()
    for v in model.get("modelVersions", []) or []:
        got = (v.get("baseModel") or "").lower()
        if want in got:
            return True
    return False


def _apply_extra_filters(items, tag_categories, tag_filter_text, base_model_value):
    required_text_tags = _parse_tag_list(tag_filter_text)
    category_tags = list(tag_categories or [])
    if not required_text_tags and not category_tags and (not base_model_value or base_model_value == "Any"):
        return list(items or [])
    out = []
    for m in items or []:
        if not _model_matches_base_model(m, base_model_value):
            continue
        if not _model_matches_tags(m, required_text_tags):
            continue
        if not _model_matches_any_tag(m, category_tags):
            continue
        out.append(m)
    return out


_CONTENT_LEVEL_ORDER = {"PG": 0, "PG-13": 1, "R": 2, "X": 3, "XXX": 4}


def _normalize_content_level(value):
    if value is None:
        return "PG"
    if isinstance(value, bool):
        return "XXX" if value else "PG"
    if isinstance(value, (int, float)):
        idx = int(value)
        for k, v in _CONTENT_LEVEL_ORDER.items():
            if v == idx:
                return k
        if idx <= 0:
            return "PG"
        if idx == 1:
            return "PG-13"
        if idx == 2:
            return "R"
        if idx == 3:
            return "X"
        return "XXX"
    if isinstance(value, str):
        raw = value.strip().upper()
        raw = raw.replace("PG13", "PG-13")
        if raw in {"SAFE", "SFW", "NONE"}:
            return "PG"
        if raw in {"NSFW"}:
            return "XXX"
        if raw in _CONTENT_LEVEL_ORDER:
            return raw
        if raw in {"MATURE", "ADULT"}:
            return "R"
        if raw in {"EXPLICIT"}:
            return "XXX"
    return "PG"


def _allowed_content_levels(levels):
    if not levels:
        return set(_CONTENT_LEVEL_ORDER.keys())
    return {_normalize_content_level(lvl) for lvl in levels if (lvl or "").strip()}


def _model_content_level(model):
    direct = model.get("nsfwLevel", None)
    if direct is not None:
        return _normalize_content_level(direct)
    direct = model.get("nsfw", None)
    if direct is not None:
        return _normalize_content_level(direct)
    max_rank = 0
    for v in model.get("modelVersions", []) or []:
        v_lvl = v.get("nsfwLevel", None)
        if v_lvl is not None:
            max_rank = max(max_rank, _CONTENT_LEVEL_ORDER.get(_normalize_content_level(v_lvl), 0))
            continue
        v_lvl = v.get("nsfw", None)
        if v_lvl is not None:
            max_rank = max(max_rank, _CONTENT_LEVEL_ORDER.get(_normalize_content_level(v_lvl), 0))
        for img in v.get("images", []) or []:
            i_lvl = img.get("nsfwLevel", None)
            if i_lvl is not None:
                max_rank = max(max_rank, _CONTENT_LEVEL_ORDER.get(_normalize_content_level(i_lvl), 0))
                continue
            i_lvl = img.get("nsfw", None)
            if i_lvl is not None:
                max_rank = max(max_rank, _CONTENT_LEVEL_ORDER.get(_normalize_content_level(i_lvl), 0))
    for k, v in _CONTENT_LEVEL_ORDER.items():
        if v == max_rank:
            return k
    return "PG"


def _model_matches_content_levels(model, levels):
    allowed = _allowed_content_levels(levels)
    has_any_flag = False
    for v in model.get("modelVersions", []) or []:
        v_lvl = v.get("nsfwLevel", None)
        if v_lvl is not None:
            has_any_flag = True
            if _normalize_content_level(v_lvl) in allowed:
                return True
        v_lvl = v.get("nsfw", None)
        if v_lvl is not None:
            has_any_flag = True
            if _normalize_content_level(v_lvl) in allowed:
                return True
        for img in v.get("images", []) or []:
            i_lvl = img.get("nsfwLevel", None)
            if i_lvl is not None:
                has_any_flag = True
                if _normalize_content_level(i_lvl) in allowed:
                    return True
            i_lvl = img.get("nsfw", None)
            if i_lvl is not None:
                has_any_flag = True
                if _normalize_content_level(i_lvl) in allowed:
                    return True
    m_lvl = model.get("nsfwLevel", None)
    if m_lvl is not None:
        has_any_flag = True
        if _normalize_content_level(m_lvl) in allowed:
            return True
    m_lvl = model.get("nsfw", None)
    if m_lvl is not None:
        has_any_flag = True
        if _normalize_content_level(m_lvl) in allowed:
            return True
    if not has_any_flag:
        return "PG" in allowed
    return False


def build_search_url(query, model_type, sort, content_levels, api_key, creator_filter, period="AllTime", use_tag=False):
    creator_active = creator_filter and creator_filter != "— All —"
    include_nsfw = creator_active or any((lvl or "").strip().upper() in ["NSFW", "PG-13", "R", "X", "XXX"] for lvl in content_levels)
    params = {
        "limit": 20,
        "sort": sort,
        "period": period,
        "nsfw": str(include_nsfw).lower(),
    }

    if model_type != "All":
        params["types"] = model_type

    if creator_active:
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


def _pick_model_preview_image_url(model: dict, allowed_levels=None):
    versions = model.get("modelVersions", []) or []
    sel_id = model.get("_civitai_selected_version_id", None)
    ordered = []
    if sel_id is not None:
        for v in versions:
            if str(v.get("id")) == str(sel_id):
                ordered.append(v)
                break
    if ordered:
        ordered += [v for v in versions if str(v.get("id")) != str(sel_id)]
    else:
        ordered = list(versions)
    for v in ordered:
        thumb = _pick_version_preview_image_url(v or {}, allowed_levels=allowed_levels)
        if thumb:
            return thumb
    return ""


def _has_thumbnail(model, allowed_levels=None):
    return bool(_pick_model_preview_image_url(model, allowed_levels=allowed_levels))


def build_gallery_data(items, allowed_levels=None):
    gallery = []
    for m in items:
        thumb = _pick_model_preview_image_url(m or {}, allowed_levels=allowed_levels)
        if thumb:
            gallery.append((thumb, m.get("name", "?")))
    return gallery


def _pick_version_preview_image_url(version: dict, allowed_levels=None):
    if not version:
        return ""
    allowed = _allowed_content_levels(allowed_levels)
    skip_types = {"video"}
    skip_ext = {".mp4", ".webm", ".gif", ".mov", ".avi"}
    for img in version.get("images", []) or []:
        if img.get("type", "image").lower() in skip_types:
            continue
        if _normalize_content_level(img.get("nsfwLevel", img.get("nsfw", None))) not in allowed:
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
        esc = _escape_html(w)
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
            f">{esc}</span>"
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
    safe = re.sub(r"(?i)\\b(href|src)\\s*=\\s*([\"'])\\s*javascript:[^\"']*\\2", r"\\1=\"#\"", safe)
    safe = re.sub(r"(?i)\\b(href|src)\\s*=\\s*([\"'])\\s*data:[^\"']*\\2", r"\\1=\"#\"", safe)
    return safe.strip()


def _has_meaningful_html(html: str) -> bool:
    if not html:
        return False
    txt = re.sub(r"<[^>]+>", "", html)
    txt = txt.replace("&nbsp;", " ").replace("\u00a0", " ")
    return bool(txt.strip())


def build_open_link_html(model, version=None):
    mid = model.get("id", "")
    if not mid:
        return ""
    vid = ""
    if version:
        vid = version.get("id") or ""
    url = f"https://civitai.com/models/{mid}" + (f"?modelVersionId={vid}" if vid else "")
    return (
        f"<a href='{url}' target='_blank' "
        "style='display:inline-flex;align-items:center;padding:3px 10px;background:#1e2d3d;border:1px solid #1d4ed8;"
        "border-radius:999px;color:#60a5fa;font-size:12px;text-decoration:none;font-weight:700;white-space:nowrap'>"
        "Open on CivitAI</a>"
    )


def get_model_header_html(model, version=None):
    if not model:
        return ""

    if version is None and model.get("modelVersions"):
        version = model["modelVersions"][0]

    stats = model.get("stats", {}) or {}
    downloads = stats.get("downloadCount", 0)
    rating = float(stats.get("rating", 0) or 0)
    ratingcnt = int(stats.get("ratingCount", 0) or 0)
    creator = _escape_html((model.get("creator") or {}).get("username", "NA"))
    modeltype_raw = model.get("type", "Other")
    modeltype = _escape_html(modeltype_raw)
    typecolor = TYPE_COLORS.get(modeltype_raw, "#374151")
    model_name = _escape_html(model.get("name", "NA"))

    stars = ""
    if ratingcnt > 0:
        stars = (
            "<span style='background:#2a2209;border:1px solid #92400e;color:#fbbf24;"
            "padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600'>"
            f"{rating:.1f} ★ ({ratingcnt:,})</span>"
        )

    open_link = build_open_link_html(model, version)

    return (
        "<div style='padding:12px 14px;font-family:sans-serif;color:#e0e0e0'>"
        "<div style='margin-bottom:10px'>"
        "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px'>"
        f"<h3 style='margin:0;color:#fff;font-size:16px;line-height:1.3;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{model_name}</h3>"
        "</div>"
        "<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap'>"
        f"<span style='background:{typecolor};color:#fff;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:700;white-space:nowrap;flex-shrink:0'>{modeltype}</span>"
        f"<span style='background:#1e2d3d;border:1px solid #1d4ed8;color:#60a5fa;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600'>{creator}</span>"
        f"<span style='background:#1a2e1a;border:1px solid #166534;color:#34d399;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600'>{downloads:,} downloads</span>"
        f"{stars}"
        f"{open_link}"
        "</div>"
        "</div>"
        "</div>"
    )


def get_model_body_html(model, version=None):
    if not model:
        return EMPTY_DETAIL

    if version is None and model.get("modelVersions"):
        version = model["modelVersions"][0]

    rawdesc = model.get("description") or ""
    if not rawdesc and version:
        rawdesc = version.get("description") or ""
    safedesc = sanitize_description_html(rawdesc)

    desc_html = (
        "<details style='margin-top:10px'>"
        "<summary style='cursor:pointer;padding:8px 12px;background:#1e2a1e;border-radius:6px;"
        "border-left:3px solid #4ade80;color:#4ade80;font-size:12px;font-weight:700;"
        "list-style:none;user-select:none'>Model description</summary>"
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

    return (
        "<div style='padding:12px 14px;font-family:sans-serif;color:#e0e0e0'>"
        "<div style='margin-bottom:10px'>"
        f"{desc_html}"
        f"{about_html}"
        f"{notes_html}"
        "</div>"
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
    creator = _escape_html((model.get("creator") or {}).get("username", "NA"))
    modeltype_raw = model.get("type", "Other")
    modeltype = _escape_html(modeltype_raw)
    typecolor = TYPE_COLORS.get(modeltype_raw, "#374151")
    tags = model.get("tags", [])[:6]
    model_name = _escape_html(model.get("name", "NA"))

    tags_html = ""
    if tags:
        tags_html = "<div style='margin-top:8px'>" + "".join(
            f"<span style='display:inline-block;padding:2px 8px;margin:2px;background:#1c1c2e;border:1px solid #374151;"
            f"border-radius:12px;color:#9ca3af;font-size:10px'>{_escape_html(t)}</span>"
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
        f"<h3 style='margin:0;color:#fff;font-size:16px;line-height:1.3;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{model_name}</h3>"
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
        tab_class = "civlens-tab active" if is_active else "civlens-tab"

        close_btn = ""
        if count > 1:
            close_btn = (
                "<span "
                "class='civlens-tab-close' "
                "title='Close tab' "
                f"onclick=\"event.stopPropagation();var el=document.getElementById('civlens-close-btn-{i}');if(el) el.click();\""
                "aria-label='Close tab'"
                "><span class='civlens-tab-close-icon'>×</span></span>"
            )

        tabs_html += (
            f"<div class='{tab_class}' "
            f"data-tab-index='{i}' "
            f"title='Search {i+1}' "
            f"onclick=\"var el=document.getElementById('civlens-switch-btn-{i}');if(el) el.click();\" "
            f"onauxclick=\"if(event.button===1){{event.preventDefault();var el=document.getElementById('civlens-close-btn-{i}');if(el) el.click();}}\""
            f"><span class='civlens-tab-icon' aria-hidden='true'></span><span class='civlens-tab-title'>Search {i+1}</span>{close_btn}</div>"
        )

    if count < MAX_TABS:
        tabs_html += (
            "<div class='civlens-tab-add' "
            "title='New tab' "
            "onclick=\"var el=document.getElementById('civlens-add-btn');if(el) el.click();\" "
            "aria-label='New tab'"
            "><span class='civlens-tab-add-icon'>+</span></div>"
        )

    return f"<div class='civlens-tabstrip'>{tabs_html}</div>"


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
    dl = (primary.get("downloadUrl") or "").strip() or None
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

def _download_job_key(panel_id):
    return str(panel_id)


def _download_job_snapshot(panel_id):
    key = _download_job_key(panel_id)
    with _DOWNLOAD_JOBS_LOCK:
        job = _DOWNLOAD_JOBS.get(key)
        return dict(job) if job else None


def _update_download_job(panel_id, **updates):
    key = _download_job_key(panel_id)
    with _DOWNLOAD_JOBS_LOCK:
        job = _DOWNLOAD_JOBS.get(key)
        if not job:
            return
        job.update(updates)


def _cancel_sleep(seconds, cancel_event):
    if not seconds:
        return
    end = time.time() + float(seconds)
    while time.time() < end:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Cancelled")
        time.sleep(min(0.2, end - time.time()))


def _download_get(url, headers, cancel_event, stream=False, timeout=(10, 5)):
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Cancelled")
    if not _is_allowed_url(url):
        raise ValueError("Blocked URL")
    r = requests.get(url, headers=headers or {}, timeout=timeout, stream=stream)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        try:
            delay = float(ra)
        except Exception:
            delay = 2.0
        _cancel_sleep(min(delay, 5.0), cancel_event)
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Cancelled")
        r = requests.get(url, headers=headers or {}, timeout=timeout, stream=stream)
    r.raise_for_status()
    return r


def poll_download(panel_id):
    job = _download_job_snapshot(panel_id)
    if not job:
        return gr.update(), gr.update(), gr.update(active=False)

    filename = job.get("filename") or ""
    status = job.get("status", "")
    if filename:
        progress_html = _render_progress_html(job.get("percent", 0), job.get("done", 0), job.get("total", 0), filename)
    else:
        progress_html = ""

    finished = bool(job.get("finished"))
    timer_update = gr.update(active=(not finished))

    key = _download_job_key(panel_id)
    with _DOWNLOAD_JOBS_LOCK:
        live = _DOWNLOAD_JOBS.get(key) or {}
        last_progress = live.get("ui_last_progress", None)
        last_status = live.get("ui_last_status", None)
        if last_progress == progress_html and last_status == status:
            return gr.update(), gr.update(), timer_update
        live["ui_last_progress"] = progress_html
        live["ui_last_status"] = status
        _DOWNLOAD_JOBS[key] = live

    return gr.update(value=progress_html), gr.update(value=status), timer_update


def _download_worker(panel_id, model, version, api_key):
    job = _download_job_snapshot(panel_id) or {}
    cancel_event = job.get("cancel_event")

    model_type = model.get("type", "Other")
    save_dir = get_model_dir(model_type)
    os.makedirs(save_dir, exist_ok=True)

    ver_id = version.get("id")
    dl_url, filename = _pick_download_url_and_name(version)
    if not filename:
        filename = f"{model.get('id','model')}_{ver_id or 'latest'}.safetensors"
    filename = _sanitize_filename(filename)

    dest = _safe_join(save_dir, filename)
    if os.path.exists(dest):
        existing = os.path.getsize(dest)
        _update_download_job(panel_id, filename=filename, done=existing, total=existing, percent=100, status=f"Already exists: {filename}", finished=True)
        return

    if not dl_url and ver_id:
        dl_url = f"{DOWNLOAD_URL}/{ver_id}"
    if not dl_url:
        _update_download_job(panel_id, filename=filename, status="No download URL found for this version.", finished=True)
        return

    headers = _get_headers(api_key)

    try:
        _update_download_job(panel_id, filename=filename, status=f"Starting download: {filename}", done=0, total=0, percent=0)
        with _download_get(dl_url, headers=headers, cancel_event=cancel_event, stream=True, timeout=(10, 5)) as r:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            _update_download_job(panel_id, total=total, done=0, percent=0)

            last_pct = -1
            last_ui_ts = 0.0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if cancel_event and cancel_event.is_set():
                        try:
                            f.close()
                        except Exception:
                            pass
                        try:
                            if os.path.exists(dest):
                                os.remove(dest)
                        except Exception:
                            pass
                        _update_download_job(panel_id, status="Download cancelled.", finished=True, percent=0, done=0, total=0)
                        return

                    if not chunk:
                        continue

                    f.write(chunk)
                    done += len(chunk)
                    pct = int((done / total) * 100.0) if total > 0 else 0
                    now = time.time()
                    if pct != last_pct or (now - last_ui_ts) > 0.8:
                        _update_download_job(panel_id, done=done, total=total, percent=pct, status=f"Downloading: {filename} ({done/1024/1024:.1f} MB)")
                        last_pct = pct
                        last_ui_ts = now

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
                    img_name = _sanitize_filename(img_name)
                    img_dest = _safe_join(save_dir, img_name)
                    if not os.path.exists(img_dest):
                        with _download_get(img_url, headers=headers, cancel_event=cancel_event, stream=True, timeout=(10, 5)) as ir:
                            with open(img_dest, "wb") as outf:
                                for chunk in ir.iter_content(chunk_size=1 << 20):
                                    if cancel_event and cancel_event.is_set():
                                        raise RuntimeError("Cancelled")
                                    if chunk:
                                        outf.write(chunk)
                        msg += f"\nPreview saved: {img_name}"
                    else:
                        msg += f"\nPreview exists: {img_name}"
                except Exception as ie:
                    msg += f"\nPreview download failed: {ie}"

        _update_download_job(panel_id, done=done, total=total, percent=100, status=msg, finished=True)
        return

    except Exception as e:
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except Exception:
            pass
        if str(e) == "Cancelled":
            _update_download_job(panel_id, status="Download cancelled.", finished=True)
        else:
            _update_download_job(panel_id, status=f"Download failed: {e}", finished=True)
        return


def start_download(search_data, version_choice, api_key, panel_id):
    items = search_data.get("items", [])
    idx = search_data.get("selected_index", 0)
    if not items or idx >= len(items):
        return "", "No model selected.", gr.update(active=False)

    model = items[idx]
    version = get_version_by_choice(model, version_choice)
    if not version:
        return "", "No version found.", gr.update(active=False)

    key = _download_job_key(panel_id)
    with _DOWNLOAD_JOBS_LOCK:
        existing = _DOWNLOAD_JOBS.get(key)
        if existing and existing.get("thread") and existing["thread"].is_alive():
            return poll_download(panel_id)

        ver_id = version.get("id")
        dl_url, filename = _pick_download_url_and_name(version)
        if not filename:
            filename = f"{model.get('id','model')}_{ver_id or 'latest'}.safetensors"
        filename = _sanitize_filename(filename)

        job = {
            "filename": filename,
            "done": 0,
            "total": 0,
            "percent": 0,
            "status": f"Starting download: {filename}",
            "finished": False,
            "cancel_event": threading.Event(),
        }
        worker = threading.Thread(target=_download_worker, args=(panel_id, model, version, api_key), daemon=True)
        job["thread"] = worker
        _DOWNLOAD_JOBS[key] = job
        worker.start()

    return _render_progress_html(0, 0, 0, filename), f"Starting download: {filename}", gr.update(active=True)


def stop_download(panel_id):
    key = _download_job_key(panel_id)
    with _DOWNLOAD_JOBS_LOCK:
        job = _DOWNLOAD_JOBS.get(key)
        if not job or job.get("finished") or not job.get("thread") or not job["thread"].is_alive():
            return "", "No active download.", gr.update(active=False)
        job["cancel_event"].set()
        job["status"] = "Stopping current download..."
    return "", "Stopping current download...", gr.update(active=True)


# =============================================================================
# UI PANELS
# =============================================================================
def make_panel_components(i, api_key_state):
    with gr.Column(visible=i == 0, elem_id=f"civlens-panel-{i}") as col:
        with gr.Group():
            gr.Markdown("Filters or Load by URL")
            with gr.Row():
                url_input = gr.Textbox(
                    label="Model URL",
                    show_label=False,
                    placeholder="Paste a CivitAI model or version URL",
                    elem_id=f"civlens-url-input-{i}",
                    scale=5,
                )
                url_btn = gr.Button("🔗 Load from URL", variant="secondary", scale=1, min_width=120, elem_id=f"civlens-url-btn-{i}")

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
                base_model = gr.Dropdown(
                    label="Base model",
                    choices=["Any", "Pony", "Illustrious", "SDXL", "SD 1.5", "SD 2.1", "Flux", "Z image Base", "Z Image turbo"],
                    value="Any",
                    scale=2,
                )
                creator_filter = gr.Dropdown(
                    label="Creator",
                    choices=creator_dropdown_choices(),
                    value="— All —",
                    scale=2,
                )

            with gr.Row():
                tag_filter = gr.Textbox(
                    label="Tags",
                    placeholder="Comma separated tags (e.g. character, clothing, cyberpunk)",
                    lines=1,
                    scale=5,
                )
                tag_categories = gr.CheckboxGroup(
                    label="Tag categories",
                    choices=["Background", "Base model", "Buildings", "Character", "Clothing", "Concept", "Poses", "Style"],
                    value=[],
                    scale=3,
                )
                content_levels = gr.CheckboxGroup(
                    label="Content rating",
                    choices=["PG", "PG-13", "R", "X", "XXX"],
                    value=["PG", "PG-13", "R", "X", "XXX"],
                    scale=3,
                )

            with gr.Row():
                search_btn = gr.Button("🔍 Load models", variant="primary", scale=4, min_width=220, elem_classes=["btn-load"])

        with gr.Group():
            gr.Markdown("Load Models and Keyword Filter")
            with gr.Row():
                query = gr.Textbox(
                    label="Keyword filter",
                    show_label=False,
                    placeholder="Filter by keyword (character name, series name...)",
                    elem_id=f"civlens-query-{i}",
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
                    elem_id=f"civlens-gallery-{i}",
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
                model_header_html = gr.HTML("")
                version_selector = gr.Dropdown(
                    label="Select the version",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False,
                )
                trigger_html = gr.HTML(build_trigger_words_html([]))
                model_body_html = gr.HTML(EMPTY_DETAIL)
                selected_url = gr.Textbox(value="", visible=False, elem_id=f"civlens-selected-url-{i}")

                gr.HTML('<hr style="border-color:#1f2937;margin:0">')

                with gr.Row():
                    download_btn = gr.Button("⬇️ Download model", variant="primary", scale=3, min_width=220, elem_classes=["btn-download"])
                    stop_btn = gr.Button("⏹️ Stop download", variant="secondary", scale=1, min_width=140)
                    send_tab_btn = gr.Button("📤 Send to new tab", variant="secondary", scale=1, min_width=170, elem_id=f"civlens-send-tab-{i}")
                dl_progress_html = gr.HTML("")

                dl_status = gr.Textbox(
                    label="Download status",
                    show_label=True,
                    interactive=False,
                    lines=3,
                    placeholder="Download status appears here.",
                )
                dl_poll_timer = gr.Timer(1.0, active=False)

        # State
        panel_id_state = gr.State(i)
        search_data = gr.State(
            {
                "items": [],
                "metadata": {},
                "all_items": [],
                "next_page": "",
                "first_page": "",
                "query": "",
                "tag_categories": [],
                "tag_filter": "",
                "base_model": "Any",
                "content_levels": ["PG", "PG-13", "R", "X", "XXX"],
                "selected_index": 0,
            }
        )

        # Events
        def on_gallery_select(evt: gr.SelectData, sd):
            items = sd.get("items", [])
            if not items or evt.index is None or evt.index >= len(items):
                return (
                    "",
                    gr.update(visible=False, interactive=False, choices=[], value=None),
                    build_trigger_words_html([]),
                    EMPTY_DETAIL,
                    "",
                    sd,
                )

            model = items[evt.index]
            versions = model.get("modelVersions", []) or []
            choices = [_version_label(v) for v in versions]
            sel_id = model.get("_civitai_selected_version_id", None)
            sel_version = None
            if sel_id is not None:
                for v in versions:
                    if str(v.get("id")) == str(sel_id):
                        sel_version = v
                        break
            if sel_version is None and versions:
                sel_version = versions[0]
            val = _version_label(sel_version) if sel_version else (choices[0] if choices else None)
            mid = model.get("id", "")
            vid = (sel_version or {}).get("id")
            sel_url = (f"https://civitai.com/models/{mid}" if mid else "") + (f"?modelVersionId={vid}" if mid and vid else "")

            sd2 = dict(sd)
            sd2["selected_index"] = int(evt.index)
            if vid is not None:
                items2 = list(items)
                m2 = dict(model)
                m2["_civitai_selected_version_id"] = vid
                items2[evt.index] = m2
                sd2["items"] = items2
                model = m2

            return (
                get_model_header_html(model, sel_version),
                gr.update(choices=choices, value=val, visible=True, interactive=len(choices) > 1),
                build_trigger_words_html(get_trigger_words_for_version(sel_version)),
                get_model_body_html(model, sel_version),
                sel_url,
                sd2,
            )

        gallery.select(
            fn=on_gallery_select,
            inputs=[search_data],
            outputs=[model_header_html, version_selector, trigger_html, model_body_html, selected_url, search_data],
        )

        def on_version_change(vc, sd):
            items = sd.get("items", [])
            idx = sd.get("selected_index", 0)
            if not items or idx >= len(items):
                return [], "", build_trigger_words_html([]), EMPTY_DETAIL, "", sd

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
            levels = sd.get("content_levels", [])

            return (
                build_gallery_data(items2, levels),
                get_model_header_html(m2, v),
                build_trigger_words_html(get_trigger_words_for_version(v)),
                get_model_body_html(m2, v),
                sel_url,
                sd2,
            )

        version_selector.change(
            fn=on_version_change,
            inputs=[version_selector, search_data],
            outputs=[gallery, model_header_html, trigger_html, model_body_html, selected_url, search_data],
        )

        def load_from_url(url, api_key, levels):
            empty_sd = {
                "items": [],
                "metadata": {},
                "all_items": [],
                "next_page": "",
                "first_page": "",
                "query": "",
                "content_levels": (levels or ["PG", "PG-13", "R", "X", "XXX"]),
                "selected_index": 0,
            }

            model_id, version_id = parse_civitai_url(url)
            if not model_id:
                return (
                    [],
                    gr.update(value="URL not recognized.", visible=True),
                    gr.update(value="", visible=False),
                    gr.update(visible=False, interactive=False),
                    "",
                    build_trigger_words_html([]),
                    EMPTY_DETAIL,
                    "",
                    empty_sd,
                )

            model, err = fetch_model_by_id(model_id, api_key)
            if err or not model:
                return (
                    [],
                    gr.update(value=(err or "Not found."), visible=True),
                    gr.update(value="", visible=False),
                    gr.update(visible=False, interactive=False),
                    "",
                    build_trigger_words_html([]),
                    EMPTY_DETAIL,
                    "",
                    empty_sd,
                )

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
                "content_levels": (levels or ["PG", "PG-13", "R", "X", "XXX"]),
                "selected_index": 0,
            }

            return (
                build_gallery_data([m2], levels),
                gr.update(value=f"Loaded: {model.get('name','?')}", visible=True),
                gr.update(value="", visible=False),
                gr.update(choices=ver_choices, value=ver_val, visible=True, interactive=len(ver_choices) > 1),
                get_model_header_html(m2, selected_ver),
                build_trigger_words_html(get_trigger_words_for_version(selected_ver)),
                get_model_body_html(m2, selected_ver),
                sel_url,
                new_sd,
            )

        url_btn.click(
            fn=load_from_url,
            inputs=[url_input, api_key_state, content_levels],
            outputs=[gallery, url_status, page_info, version_selector, model_header_html, trigger_html, model_body_html, selected_url, search_data],
        )
        url_input.submit(
            fn=load_from_url,
            inputs=[url_input, api_key_state, content_levels],
            outputs=[gallery, url_status, page_info, version_selector, model_header_html, trigger_html, model_body_html, selected_url, search_data],
        )

        def do_search(q, mt, srt, levels, api_key, creator, per, cats, tag_text, bm, sd):
            items, meta, next_page, first_page = search_first_page(q, mt, srt, levels, api_key, creator, per)
            items = [m for m in items if _model_matches_content_levels(m, levels)]
            visible_items = [m for m in items if _has_thumbnail(m, levels)]
            creator_active = creator and creator != "— All —"
            all_loaded = list(visible_items)
            if creator_active and next_page:
                headers = _get_headers(api_key)
                seen = {m.get("id") for m in all_loaded if m.get("id") is not None}
                pages = 1
                while next_page:
                    pages += 1
                    items2, meta2, next2 = _fetch_url(next_page, headers)
                    items2 = [m for m in items2 if _model_matches_content_levels(m, levels)]
                    visible2 = [m for m in items2 if _has_thumbnail(m, levels)]
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

            filtered_visible = _apply_extra_filters(visible_items, cats, tag_text, bm)
            filtered_all = _apply_extra_filters(all_loaded, cats, tag_text, bm)

            if creator_active:
                total = meta.get("totalItems", len(filtered_all))
                page_lbl = f"Loaded {len(filtered_all)} of {total} results" if filtered_all else "No results found."
            else:
                total = meta.get("totalItems", len(filtered_visible))
                page_lbl = f"Page 1: {len(filtered_visible)} of {total} results" if filtered_visible else "No results found."

            new_sd = {
                "items": (filtered_all if creator_active else filtered_visible),
                "metadata": meta,
                "all_items": (filtered_all if creator_active else filtered_visible),
                "next_page": ("" if creator_active else next_page),
                "first_page": first_page,
                "query": q,
                "tag_categories": (cats or []),
                "tag_filter": (tag_text or ""),
                "base_model": (bm or "Any"),
                "content_levels": (levels or ["PG", "PG-13", "R", "X", "XXX"]),
                "selected_index": 0,
            }

            return (
                build_gallery_data(filtered_all if creator_active else filtered_visible, levels),
                gr.update(value=page_lbl, visible=True),
                gr.update(value="", visible=False),
                "",
                gr.update(visible=False, interactive=False, choices=[], value=None),
                build_trigger_words_html([]),
                EMPTY_DETAIL,
                "",
                new_sd,
            )

        search_btn.click(
            fn=do_search,
            inputs=[query, model_type, sort, content_levels, api_key_state, creator_filter, period, tag_categories, tag_filter, base_model, search_data],
            outputs=[gallery, page_info, url_status, model_header_html, version_selector, trigger_html, model_body_html, selected_url, search_data],
        )
        query.submit(
            fn=do_search,
            inputs=[query, model_type, sort, content_levels, api_key_state, creator_filter, period, tag_categories, tag_filter, base_model, search_data],
            outputs=[gallery, page_info, url_status, model_header_html, version_selector, trigger_html, model_body_html, selected_url, search_data],
        )

        def do_next(sd, api_key):
            next_url = sd.get("next_page", "")
            if not next_url:
                levels = sd.get("content_levels", [])
                return build_gallery_data(sd.get("items", []), levels), gr.update(value="No more pages.", visible=True), "", gr.update(visible=False, interactive=False, choices=[], value=None), build_trigger_words_html([]), EMPTY_DETAIL, "", sd

            headers = _get_headers(api_key)
            items, meta, next2 = _fetch_url(next_url, headers)
            levels = sd.get("content_levels", [])
            items = [m for m in items if _model_matches_content_levels(m, levels)]
            visible_items = [m for m in items if _has_thumbnail(m, levels)]
            visible_items = _apply_extra_filters(visible_items, sd.get("tag_categories"), sd.get("tag_filter"), sd.get("base_model"))
            all_items = (sd.get("all_items") or []) + visible_items
            total = meta.get("totalItems", 0)
            page_lbl = f"{len(all_items)} of {total} loaded" if total else f"{len(all_items)} loaded"

            new_sd = dict(sd)
            new_sd.update({"items": visible_items, "metadata": meta, "all_items": all_items, "next_page": next2, "selected_index": 0})
            return build_gallery_data(visible_items, levels), gr.update(value=page_lbl, visible=True), "", gr.update(visible=False, interactive=False, choices=[], value=None), build_trigger_words_html([]), EMPTY_DETAIL, "", new_sd

        next_btn.click(
            fn=do_next,
            inputs=[search_data, api_key_state],
            outputs=[gallery, page_info, model_header_html, version_selector, trigger_html, model_body_html, selected_url, search_data],
        )

        def do_prev(sd, api_key):
            first_url = sd.get("first_page", "")
            if not first_url:
                levels = sd.get("content_levels", [])
                return build_gallery_data(sd.get("items", []), levels), gr.update(value="Already on first page.", visible=True), "", gr.update(visible=False, interactive=False, choices=[], value=None), build_trigger_words_html([]), EMPTY_DETAIL, "", sd

            headers = _get_headers(api_key)
            items, meta, next2 = _fetch_url(first_url, headers)
            levels = sd.get("content_levels", [])
            items = [m for m in items if _model_matches_content_levels(m, levels)]
            visible_items = [m for m in items if _has_thumbnail(m, levels)]
            visible_items = _apply_extra_filters(visible_items, sd.get("tag_categories"), sd.get("tag_filter"), sd.get("base_model"))
            total = meta.get("totalItems", len(visible_items))
            page_lbl = f"Page 1: {len(visible_items)} of {total} results" if visible_items else "No results."

            new_sd = dict(sd)
            new_sd.update({"items": visible_items, "metadata": meta, "all_items": visible_items, "next_page": next2, "selected_index": 0})
            return build_gallery_data(visible_items, levels), gr.update(value=page_lbl, visible=True), "", gr.update(visible=False, interactive=False, choices=[], value=None), build_trigger_words_html([]), EMPTY_DETAIL, "", new_sd

        prev_btn.click(
            fn=do_prev,
            inputs=[search_data, api_key_state],
            outputs=[gallery, page_info, model_header_html, version_selector, trigger_html, model_body_html, selected_url, search_data],
        )

        def do_refine(q, sd, api_key):
            all_items = sd.get("all_items", []) or []
            levels = sd.get("content_levels", [])
            if not all_items:
                items, meta, next_page, first_page = search_first_page(q, "All", "Most Downloaded", levels or ["PG", "PG-13", "R", "X", "XXX"], api_key, "— All —", "AllTime")
                items = [m for m in items if _model_matches_content_levels(m, levels)]
                visible_items = [m for m in items if _has_thumbnail(m, levels)]
                all_items = visible_items
                sd = dict(sd)
                sd.update({"items": visible_items, "metadata": meta, "all_items": visible_items, "next_page": next_page, "first_page": first_page, "selected_index": 0})

            if not q.strip():
                matched = [m for m in all_items if _model_matches_content_levels(m, levels)]
            else:
                qq = q.strip().lower()
                matched = [m for m in all_items if _matches_query(m, qq) and _model_matches_content_levels(m, levels)]
            matched = _apply_extra_filters(matched, sd.get("tag_categories"), sd.get("tag_filter"), sd.get("base_model"))

            sd2 = dict(sd)
            sd2["items"] = matched
            sd2["selected_index"] = 0

            page_lbl = f"{len(matched)} matches from {len(all_items)} cached"

            return (
                build_gallery_data(matched, levels),
                gr.update(value=page_lbl, visible=True),
                gr.update(value="", visible=False),
                "",
                gr.update(visible=False, interactive=False, choices=[], value=None),
                build_trigger_words_html([]),
                EMPTY_DETAIL,
                "",
                sd2,
            )

        refine_btn.click(
            fn=do_refine,
            inputs=[query, search_data, api_key_state],
            outputs=[gallery, page_info, url_status, model_header_html, version_selector, trigger_html, model_body_html, selected_url, search_data],
        )

        clear_targets = [
            url_input,
            url_status,
            query,
            gallery,
            page_info,
            model_header_html,
            version_selector,
            trigger_html,
            model_body_html,
            selected_url,
            search_data,
        ]

        def clear_tab():
            empty_sd = {
                "items": [],
                "metadata": {},
                "all_items": [],
                "next_page": "",
                "first_page": "",
                "query": "",
                "content_levels": ["PG", "PG-13", "R", "X", "XXX"],
                "selected_index": 0,
            }
            return (
                "",
                gr.update(value="", visible=False),
                "",
                [],
                gr.update(value="", visible=False),
                "",
                gr.update(choices=[], value=None, visible=False, interactive=False),
                build_trigger_words_html([]),
                EMPTY_DETAIL,
                "",
                empty_sd,
            )

        download_btn.click(
            fn=start_download,
            inputs=[search_data, version_selector, api_key_state, panel_id_state],
            outputs=[dl_progress_html, dl_status, dl_poll_timer],
        )
        stop_btn.click(
            fn=stop_download,
            inputs=[panel_id_state],
            outputs=[dl_progress_html, dl_status, dl_poll_timer],
        )
        dl_poll_timer.tick(
            fn=poll_download,
            inputs=[panel_id_state],
            outputs=[dl_progress_html, dl_status, dl_poll_timer],
        )

    return col, creator_filter, clear_tab, clear_targets


# =============================================================================
# CSS
# =============================================================================
STYLE_PATH = os.path.join(EXTENSION_DIR, "civlens.css")
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

    with gr.Blocks(analytics_enabled=False, css=CSS, elem_id="civlens-ext") as civitai_tab:
        api_key_state = gr.State(settings.get("api_key", ""))

        tab_count = gr.State(1)
        active_tab = gr.State(0)

        with gr.Tabs():
            with gr.TabItem("CivLens"):
                tab_bar = gr.HTML(render_tab_bar(1, 0))

                # Hidden-but-rendered controls (kept for JS-driven switching)
                with gr.Row(elem_classes=["civlens-hidden-controls"]):
                    add_btn = gr.Button("add", elem_id="civlens-add-btn", visible=False)
                    close_btns = [
                        gr.Button(f"close-{i}", elem_id=f"civlens-close-btn-{i}", visible=False) for i in range(MAX_TABS)
                    ]
                    switch_btns = [
                        gr.Button(f"switch-{i}", elem_id=f"civlens-switch-btn-{i}", visible=False) for i in range(MAX_TABS)
                    ]

                panels = [make_panel_components(i, api_key_state) for i in range(MAX_TABS)]
                panel_cols = [p[0] for p in panels]
                creator_filters = [p[1] for p in panels]
                panel_clear_fns = [p[2] for p in panels]
                panel_clear_targets = [p[3] for p in panels]
                clear_outputs = [c for targets in panel_clear_targets for c in targets]

                def _vis_updates(count, active):
                    return [gr.update(visible=(j < count and j == active)) for j in range(MAX_TABS)]

                def _noop_clear_updates():
                    return [gr.update() for _ in range(len(clear_outputs))]

                def _clear_updates_for(idx):
                    updates = []
                    for j in range(MAX_TABS):
                        if j == idx:
                            updates += list(panel_clear_fns[j]())
                        else:
                            updates += [gr.update() for _ in range(len(panel_clear_targets[j]))]
                    return updates

                def do_add(count, active):
                    if count >= MAX_TABS:
                        return [count, active, render_tab_bar(count, active)] + _vis_updates(count, active) + _noop_clear_updates()
                    new_count = count + 1
                    new_active = count
                    return [new_count, new_active, render_tab_bar(new_count, new_active)] + _vis_updates(new_count, new_active) + _noop_clear_updates()

                def do_close(idx, count, active):
                    if count <= 1:
                        return [count, active, render_tab_bar(count, active)] + _vis_updates(count, active) + _noop_clear_updates()
                    new_count = count - 1
                    new_active = active
                    if active == idx:
                        new_active = min(idx, new_count - 1)
                    elif active > idx:
                        new_active = active - 1
                    return [new_count, new_active, render_tab_bar(new_count, new_active)] + _vis_updates(new_count, new_active) + _clear_updates_for(idx)

                def do_switch(idx, count, active):
                    if idx >= count:
                        return [count, active, render_tab_bar(count, active)] + _vis_updates(count, active) + _noop_clear_updates()
                    return [count, idx, render_tab_bar(count, idx)] + _vis_updates(count, idx) + _noop_clear_updates()

                shared_outputs = [tab_count, active_tab, tab_bar] + panel_cols + clear_outputs

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
                        return [gr.update(), "No creator entered."] + [gr.update() for _ in creator_filters]
                    s = load_settings()
                    favs = s.get("favorite_creators", [])
                    if username not in favs:
                        favs.append(username)
                    s["favorite_creators"] = favs
                    save_settings(s)
                    creator_choices = ["— All —"] + favs
                    creator_updates = [gr.update(choices=creator_choices, value="— All —") for _ in creator_filters]
                    return [gr.update(choices=favs, value=None), f"Added: {username}"] + creator_updates

                add_creator_btn.click(
                    fn=add_creator,
                    inputs=[new_favorite_input],
                    outputs=[favorites_list, creator_status] + creator_filters,
                )

                def remove_creator(username):
                    if not username:
                        return [gr.update(), "No creator selected."] + [gr.update() for _ in creator_filters]
                    s = load_settings()
                    favs = [f for f in s.get("favorite_creators", []) if f != username]
                    s["favorite_creators"] = favs
                    save_settings(s)
                    creator_choices = ["— All —"] + favs
                    creator_updates = [gr.update(choices=creator_choices, value="— All —") for _ in creator_filters]
                    return [gr.update(choices=favs, value=None), f"Removed: {username}"] + creator_updates

                remove_creator_btn.click(
                    fn=remove_creator,
                    inputs=[favorites_list],
                    outputs=[favorites_list, creator_status] + creator_filters,
                )

    return [(civitai_tab, "CivLens", "civlens")]


script_callbacks.on_ui_tabs(on_ui_tabs)
