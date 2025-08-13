import os
import time
import re
import urllib.parse
import requests
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from reddit_tldr import (
    fetch_submission_data,
    verify_reddit_credentials,
)

# Always load the .env that sits next to this file (works regardless of CWD)
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

# --- OpenRouter config ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free")

SITE_URL = os.getenv("SITE_URL", "")
SITE_NAME = os.getenv("SITE_NAME", "")

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env next to app.py")

# Init client
llm = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)

app = Flask(__name__, template_folder=str(Path(__file__).with_name("templates")))

# ---------- Rate limiting ----------
# Default: 20 requests/min per client IP across all routes
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["20 per minute"],
)

# ---------- Simple in-memory cache (30 min TTL) ----------
_CACHE_TTL_SECS = 30 * 60
# key -> (ts, data, summary)
_CACHE: dict[str, tuple[float, dict, str]] = {}

def _cache_key(url: str, sort: str, max_comments: int, model: str) -> str:
    return f"{model}|{sort}|{max_comments}|{url}"

def _get_cache(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, data, summary = entry
    if (time.time() - ts) > _CACHE_TTL_SECS:
        _CACHE.pop(key, None)
        return None
    return data, summary

def _set_cache(key: str, data: dict, summary: str):
    _CACHE[key] = (time.time(), data, summary)
    # Simple size control
    if len(_CACHE) > 256:
        # drop oldest ~64 entries
        for i, k in enumerate(list(_CACHE.keys())):
            if i >= 64:
                break
            _CACHE.pop(k, None)

# ---------- Input guardrails ----------
_ALLOWED_SORTS = {"top", "new", "controversial", "best", "hot", "old", "q&a", "qa"}
_MIN_COMMENTS = 50
_MAX_COMMENTS = 1200

def _clamp_comments(n: int) -> int:
    if n < _MIN_COMMENTS: return _MIN_COMMENTS
    if n > _MAX_COMMENTS: return _MAX_COMMENTS
    return n

def _validate_sort(sort: str) -> str:
    s = (sort or "top").strip().lower()
    if s not in _ALLOWED_SORTS:
        raise ValueError(f"Invalid sort '{sort}'. Allowed: {', '.join(sorted(_ALLOWED_SORTS))}.")
    return s

# ---------- URL handling ----------

# Accept reddit app short path: .../s/<token> (must be at the end of the path)
S_PATH_RE = re.compile(r"/s/[A-Za-z0-9]+/?$", re.IGNORECASE)

def resolve_reddit_s_short(url: str, timeout: int = 7) -> str:
    """
    Follows redirects for reddit.com .../s/<token> short links and returns
    the final canonical reddit URL (usually /comments/<id>/...).
    Returns the input URL on any failure.
    """
    try:
        # Ensure scheme
        if not re.match(r"^\w+://", url):
            url = "https://" + url

        u = urllib.parse.urlparse(url)
        if "reddit.com" not in u.netloc.lower() or not S_PATH_RE.search(u.path):
            return url  # not an /s/ link; leave unchanged

        headers = {
            "User-Agent": os.getenv("REDDIT_USER_AGENT", "reddit-tldr-bot (resolve s-link)")
        }

        # Try HEAD first (cheap)
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        final_url = r.url

        # Fall back to GET if HEAD isn't allowed / didn't redirect
        if (final_url == url) or (r.status_code in (405, 403)) or S_PATH_RE.search(urllib.parse.urlparse(final_url).path or ""):
            g = requests.get(url, allow_redirects=True, timeout=timeout, headers=headers)
            final_url = g.url

        # Normalize host and strip query/fragment
        uu = urllib.parse.urlparse(final_url)
        path_only = uu.path or "/"
        if "redd.it" in uu.netloc.lower():
            return f"https://{uu.netloc}{path_only}"
        return f"https://www.reddit.com{path_only}"
    except Exception:
        return url

def normalize_reddit_url(raw: str) -> str:
    """
    Accept reddit.com (and subdomains), redd.it, and reddit app short-path links:
      - reddit.com ... /comments/...   -> normalize to https://www.reddit.com/<path>
      - reddit.com ... /s/<token>      -> allowed; will be resolved to /comments/... later
      - redd.it/<id>                   -> pass-through
    Strips query/fragment.
    """
    if not raw:
        raise ValueError("Please paste a Reddit post URL.")

    raw = raw.strip()
    # Add scheme if missing
    if "://" not in raw:
        raw = "https://" + raw

    u = urlparse(raw)
    host = (u.netloc or "").lower()

    valid_hosts = (
        "reddit.com",
        "www.reddit.com",
        "old.reddit.com",
        "np.reddit.com",
        "m.reddit.com",
        "new.reddit.com",
        "amp.reddit.com",
        "redd.it",
        "www.redd.it",
    )
    if not any(host == h or host.endswith("." + h) for h in valid_hosts):
        raise ValueError("That doesn't look like a Reddit URL.")

    path = u.path or "/"

    # redd.it shortlinks are OK (pass through, canonical host preserved)
    if "redd.it" in host:
        return urlunparse(("https", host, path, "", "", ""))

    # reddit.com variants:
    # allow either a comments permalink OR an /s/<token> app short path
    is_comments = "/comments/" in path
    is_s_short = bool(S_PATH_RE.search(path))
    if not (is_comments or is_s_short):
        raise ValueError("Please paste a direct post link that contains /comments/, /s, or a redd.it shortlink.")

    # Normalize to www.reddit.com and strip query/fragment
    return urlunparse(("https", "www.reddit.com", path, "", "", ""))

# -------- LLM helpers (low volatility) --------
_SYSTEM_STYLE = (
    "You write single-paragraph, family-friendly TL;DRs.\n"
    "Summarize Reddit COMMENTS only. Keep it 3–6 sentences.\n"
    "Include overall sentiment and main viewpoints. Paraphrase any profanity.\n"
    "No lists or headings. Keep tone neutral and concise."
)

def _call_llm_with_retry(messages, attempts: int = 3, timeout: int = 60):
    last = None
    headers = {}
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-Title"] = SITE_NAME

    for i in range(attempts):
        try:
            r = llm.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                temperature=0,
                top_p=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                extra_body={"seed": 42},  # some models honor this; harmless if ignored
                extra_headers=headers or None,
                timeout=timeout,
            )
            return r
        except Exception as e:
            last = e
            # brief backoff then retry
            sleep_s = 0.6 * (i + 1)
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                sleep_s = 0.8 * (i + 1)
            time.sleep(sleep_s)
    raise last

def summarize_with_openrouter(paragraph: str) -> str:
    r = _call_llm_with_retry(
        [
            {"role": "system", "content": _SYSTEM_STYLE},
            {"role": "user", "content": paragraph},
        ],
        timeout=75,
    )
    return (r.choices[0].message.content or "").strip()

def _summarize_chunk(text: str) -> str:
    r = _call_llm_with_retry(
        [
            {"role": "system", "content": _SYSTEM_STYLE},
            {"role": "user", "content": f"Chunk:\n\n{text}"},
        ],
        timeout=60,
    )
    return (r.choices[0].message.content or "").strip()

# ---- smarter, sentence-aware chunking ----
def _split_sentences(text: str) -> list[str]:
    """
    Best-effort sentence splitter without heavy deps.
    Falls back to naive '.' split; keeps punctuation.
    """
    import re
    protected = re.sub(
        r"\b(e\.g|i\.e|mr|mrs|dr|ms|jr|sr|u\.s|u\.k|etc)\.",
        lambda m: m.group(0).replace(".", "§"),
        text,
        flags=re.I,
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", protected)
    sents = []
    for p in parts:
        s = p.replace("§", ".").strip()
        if s:
            if s[-1] not in ".!?":
                s += "."
            sents.append(s)
    if not sents:
        sents = [s.strip() + "." for s in text.split(".") if s.strip()]
    return sents

def _chunk_by_chars(sents: list[str], chunk_chars: int = 3000) -> list[str]:
    chunks, cur, n = [], [], 0
    for s in sents:
        if n + len(s) + 1 > chunk_chars and cur:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def summarize_map_reduce(paragraph: str, *, chunk_chars: int = 3000) -> str:
    sents = _split_sentences(paragraph)
    parts = _chunk_by_chars(sents, chunk_chars=chunk_chars)

    if len(parts) <= 1:
        return summarize_with_openrouter(paragraph)

    # MAP: TL;DR each chunk
    mini = []
    for p in parts:
        mini.append(_summarize_chunk(p))

    # REDUCE: combine mini TL;DRs into ONE final TL;DR paragraph
    reduce_text = " ".join(mini)
    return summarize_with_openrouter(reduce_text)

# -------- Routes --------
@app.get("/health")
@limiter.exempt
def health():
    try:
        r = llm.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "system", "content": "ping"}],
            temperature=0,
            top_p=0.1,
            extra_body={"seed": 42},
        )
        ok = bool(r.choices[0].message.content)
        return jsonify({"llm_ok": ok, "model": OPENROUTER_MODEL}), 200
    except Exception as e:
        return jsonify({"llm_ok": False, "error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
@limiter.limit("20 per minute")   # tighter cap on the expensive summarize endpoint
def index():
    # Verify Reddit creds; show clear message if 401/mismatch/etc.
    try:
        verify_reddit_credentials()
    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            title=None,
            subreddit=None,
            selftext=None,
            image_url=None,
            video_url=None,
            sentiment=None,
        )

    # Defaults for form persistence & template safety
    sort = "top"
    max_comments = 500
    summary = None
    data = {}
    error = None
    url = None  # so we can reference after POST

    if request.method == "POST":
        # Guardrails
        try:
            sort = _validate_sort(request.form.get("sort"))
        except ValueError as ve:
            return render_template("index.html", summary=None, error=str(ve))

        raw_url = (request.form.get("url") or "").strip()
        try:
            max_comments = int(request.form.get("max_comments") or 500)
        except ValueError:
            max_comments = 500
        max_comments = _clamp_comments(max_comments)

        # Robust URL handling + caching + timeout safeguards
        try:
            url = normalize_reddit_url(raw_url)
            # If it's an /s/<token> app share link, resolve it to /comments/ permalink
            parsed = urlparse(url)
            if "reddit.com" in parsed.netloc.lower() and S_PATH_RE.search(parsed.path or ""):
                url = resolve_reddit_s_short(url)
        except ValueError as ve:
            error = str(ve)
            return render_template("index.html", summary=None, error=error)

        cache_key = _cache_key(url, sort, max_comments, OPENROUTER_MODEL)
        cached = _get_cache(cache_key)
        if cached:
            data, summary = cached
        else:
            try:
                data = fetch_submission_data(url, sort=sort, max_comments=max_comments)
            except Exception as e:
                msg = str(e)
                if "timed out" in msg.lower() or "timeout" in msg.lower():
                    error = "Reddit is taking too long to respond. Please try again or lower max comments."
                else:
                    error = msg
                return render_template("index.html", summary=None, error=error)

            try:
                summary = summarize_map_reduce(data["paragraph"])
            except Exception as e:
                msg = str(e)
                if "timed out" in msg.lower() or "timeout" in msg.lower():
                    error = "The language model timed out while summarizing. Please retry."
                else:
                    error = msg
                return render_template("index.html", summary=None, error=error)

            _set_cache(cache_key, data, summary)

    current_url = url if request.method == "POST" else None

    payload = {
        "summary": summary,
        "error": error,
        "title": data.get("title") if data else None,
        "subreddit": data.get("subreddit") if data else None,
        "selftext": data.get("selftext") if data else None,
        "image_url": data.get("image_url") if data else None,
        "video_url": data.get("video_url") if data else None,
        "sentiment": data.get("sentiment") if data else None,
        "current_sort": sort,
        "current_max_comments": max_comments,
        "current_url": current_url,
    }
    return render_template("index.html", **payload)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5057, debug=True)
