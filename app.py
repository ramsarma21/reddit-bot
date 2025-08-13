import os
import time
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
_CACHE: dict[str, tuple[float, dict, str]] = {}  # key -> (ts, data, summary)

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

# ---------- URL handling ----------
def normalize_reddit_url(raw: str) -> str:
    """
    Accepts reddit.com / old.reddit.com / m.reddit.com / np.reddit.com / redd.it links.
    Normalizes to https://www.reddit.com/<path> (strips query/fragment).
    Leaves redd.it shortlinks as-is (PRAW can resolve).
    Raises ValueError for non-Reddit URLs or obviously invalid forms.
    """
    if not raw:
        raise ValueError("Please paste a Reddit post URL.")

    raw = raw.strip()
    # Add scheme if missing
    if "://" not in raw:
        raw = "https://" + raw

    u = urlparse(raw)
    host = (u.netloc or "").lower()

    # Accept hosts
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

    # If redd.it shortlink, pass through (can't normalize path easily here)
    if "redd.it" in host:
        return urlunparse(("https", host, u.path, "", "", ""))

    # For reddit.com variants, normalize host and strip query/fragment
    path = u.path or "/"
    # Ensure it's a post (has /comments/); if not, still let PRAW try, but annotate
    if "/comments/" not in path:
        # Often users paste a subreddit or user URL; give a clear hint
        # but still let PRAW attempt (may handle gallery or crosspost).
        pass

    return urlunparse(("https", "www.reddit.com", path, "", "", ""))

# -------- LLM helpers (TL;DR-only) --------
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
                extra_headers=headers or None,
                timeout=timeout,
            )
            return r
        except Exception as e:
            last = e
            # Expose timeout hint in error flow
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                # brief backoff then retry
                time.sleep(0.8 * (i + 1))
            else:
                time.sleep(0.6 * (i + 1))
    raise last

def summarize_with_openrouter(paragraph: str) -> str:
    # TL;DR ONLY — no lists, no extras
    prompt = (
        "You are summarizing the COMMENTS under a Reddit post.\n"
        "Return ONLY a single TL;DR paragraph (about 3–6 sentences) covering overall sentiment, "
        "major viewpoints, and key takeaways. Do NOT include bullet points or numbered lists. "
        "STYLE: clean, concise, family-friendly. You MAY consider profane/toxic comments for analysis, "
        "but DO NOT reproduce profanity or slurs; paraphrase euphemistically.\n\n"
        + paragraph
    )
    r = _call_llm_with_retry([{"role": "user", "content": prompt}], timeout=75)
    return r.choices[0].message.content.strip()

def _summarize_chunk(text: str) -> str:
    # Chunk TL;DR used in map step
    prompt = (
        "Summarize THIS CHUNK of Reddit COMMENTS as a single concise TL;DR paragraph "
        "(2–4 sentences). No lists. Family-friendly; paraphrase any profanities.\n\n" + text
    )
    r = _call_llm_with_retry([{"role": "user", "content": prompt}], timeout=60)
    return r.choices[0].message.content.strip()

def summarize_map_reduce(paragraph: str, *, chunk_chars: int = 6000) -> str:
    # Split paragraph into near-sentence chunks
    parts, cur, total = [], [], 0
    for sent in paragraph.split(". "):
        s = (sent + ". ").strip()
        if total + len(s) > chunk_chars and cur:
            parts.append("".join(cur)); cur, total = [], 0
        cur.append(s); total += len(s)
    if cur: parts.append("".join(cur))

    if len(parts) <= 1:
        return summarize_with_openrouter(paragraph)

    # MAP: TL;DR each chunk
    mini = []
    for p in parts:
        mini.append(_summarize_chunk(p))

    # REDUCE: combine mini TL;DRs into ONE final TL;DR paragraph
    reduce_prompt = (
        "Combine the following chunk TL;DRs into ONLY one final TL;DR paragraph "
        "(about 4–7 sentences) that captures the overall sentiment, main positions, "
        "and key takeaways from the comments. Do NOT include bullets or lists. "
        "Family-friendly; do not reproduce profanity—paraphrase euphemistically.\n\n"
        + "\n\n".join(mini)
    )
    r = _call_llm_with_retry([{"role": "user", "content": reduce_prompt}], timeout=75)
    return r.choices[0].message.content.strip()

# -------- Routes --------
@app.get("/health")
@limiter.exempt
def health():
    try:
        r = llm.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": "ping"}],
        )
        ok = bool(r.choices[0].message.content)
        return jsonify({"llm_ok": ok, "model": OPENROUTER_MODEL}), 200
    except Exception as e:
        return jsonify({"llm_ok": False, "error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
@limiter.limit("5 per minute")   # tighter cap on the expensive summarize endpoint
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

    if request.method == "POST":
        sort = (request.form.get("sort") or "top").strip().lower()
        raw_url = (request.form.get("url") or "").strip()
        try:
            max_comments = int(request.form.get("max_comments") or 500)
        except ValueError:
            max_comments = 500

        # Robust URL handling + caching + timeout safeguards
        try:
            url = normalize_reddit_url(raw_url)
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
    }
    return render_template("index.html", **payload)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5057, debug=True)
