import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor

import requests
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
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["20 per minute"])

# ---------- Simple in-memory cache (30 min TTL) ----------
_CACHE_TTL_SECS = 30 * 60
# key -> (ts, data, final_tldr, pos_summary, neu_summary, neg_summary)
_CACHE: dict[str, tuple[float, dict, str, str, str, str]] = {}

def _cache_key(url: str, sort: str, model: str) -> str:
    return f"{model}|{sort}|{url}"

def _get_cache(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, data, summary, pos_s, neu_s, neg_s = entry
    if (time.time() - ts) > _CACHE_TTL_SECS:
        _CACHE.pop(key, None)
        return None
    return data, summary, pos_s, neu_s, neg_s

def _set_cache(key: str, data: dict, summary: str, pos_s: str, neu_s: str, neg_s: str):
    _CACHE[key] = (time.time(), data, summary, pos_s, neu_s, neg_s)
    if len(_CACHE) > 256:
        for i, k in enumerate(list(_CACHE.keys())):
            if i >= 64:
                break
            _CACHE.pop(k, None)

# ---------- URL handling with reddit app /s support ----------
def _resolve_reddit_redirect(url: str) -> str:
    try:
        r = requests.head(url, allow_redirects=True, timeout=6)
        final = r.url
        if not final or final == url:
            r = requests.get(url, allow_redirects=True, timeout=8)
            final = r.url
        return final or url
    except Exception:
        return url

def normalize_reddit_url(raw: str) -> str:
    """
    Accept reddit.com (subdomains), redd.it, and reddit app /s/<token> links.
    Normalize to https://www.reddit.com/<comments path>.
    """
    if not raw:
        raise ValueError("Please paste a Reddit post URL.")

    raw = raw.strip()
    if "://" not in raw:
        raw = "https://" + raw

    u = urlparse(raw)
    host = (u.netloc or "").lower()
    valid_hosts = (
        "reddit.com","www.reddit.com","old.reddit.com","np.reddit.com","m.reddit.com",
        "new.reddit.com","amp.reddit.com","redd.it","www.redd.it",
    )
    if not any(host == h or host.endswith("." + h) for h in valid_hosts):
        raise ValueError("That doesn't look like a Reddit URL.")

    # redd.it shortlinks or /s/ links → resolve
    if "redd.it" in host or "/s/" in (u.path or ""):
        final = _resolve_reddit_redirect(raw)
        u2 = urlparse(final)
        if "reddit.com" not in u2.netloc:
            return final
        return urlunparse(("https", "www.reddit.com", u2.path, "", "", ""))

    if "/comments/" not in (u.path or "/"):
        raise ValueError("Please paste a direct post link that contains /comments/ (or use a redd.it or /s/ link).")

    return urlunparse(("https", "www.reddit.com", u.path, "", "", ""))

# -------- LLM helpers --------
_SYSTEM_GROUP = (
    "You are an expert in the topic at hand (identify it). You don't make it sound like a report, but you know what you are talking about."
    "You are helping summarize Reddit comments. Ignore memes, one-liners, and jokey asides "
    "unless they meaningfully reflect sentiment. Keep 5–7 sentences, factual and specific. "
    "Prefer concrete claims, reasons, and recurring themes; drop usernames/links."
    "Ignore replies that do not value to the original comment"
    "Write in a way that it appears as just a regular tl;dr and does not include information that the reader does not need to see. Just include important points from that sentiment group. Make it clean, do not user numbers, make it a paragraph"
    "Do not use quotes verbatim, parahprase them"

)

_SYSTEM_FINAL = (
    "You are an expert in the topic at hand (identify it). You don't make it sound like a report, but you know what you are talking about."
    "Write a single, coherent paragraph (8–10 sentences) that summarizes discussion across sentiment groups in a way that readers can find out all the points mentioned in the comments with just a single read of the tl;dr. "
    "Open by reporting the largest distribution but not explicitly (e.g., 'mostly positive'), "
    "then synthesize consistent the most important and key points from all groups, in a way that all important points are mentioned. Do not contradict the distribution; "
    "Write in a way that it appears as just a regular tl;dr and does not include information that the reader does not need to see. Just include important points from that sentiment group. Make it clean, do not user numbers, make it a paragraph"
    "Do not use quotes verbatim, parahprase them"
    "if neutral dominates, say the tone is mixed/neutral. Avoid bullet lists and headings."
    "Do not use quotes verbatim, parahprase them"

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
                temperature=0.2,
                top_p=0.2,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                extra_body={"seed": 42},
                extra_headers=headers or None,
                timeout=timeout,
            )
            return r
        except Exception as e:
            last = e
            time.sleep(0.6 * (i + 1))
    raise last

def _enforce_single_paragraph(text: str, max_sentences: int = 10) -> str:
    import re as _re
    t = text or ""
    t = _re.sub(r"(\*\*|__|\*|[#>\-•]+)\s*", " ", t)
    t = _re.sub(r"\b(First|Second|Third|Fourth|Fifth)\b[^:]{0,40}:\s*", "", t, flags=_re.I)
    t = _re.sub(r"\s*\n+\s*", " ", t)
    t = _re.sub(r"\s{2,}", " ", t).strip()
    sents = _re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", t) if t else []
    if not sents:
        sents = [s.strip() + "." for s in t.split(".") if s.strip()]
    cleaned = " ".join(sents[:max_sentences]).strip()
    cleaned = _re.sub(r"\s{2,}", " ", cleaned).strip(" -–")
    return cleaned

def _summarize_group(label: str, texts: list[str]) -> str:
    # tighter cap to keep calls snappy
    blob = " ".join(texts)[:4000]
    if not blob.strip():
        return ""
    r = _call_llm_with_retry(
        [
            {"role": "system", "content": _SYSTEM_GROUP},
            {"role": "user", "content": f"Sentiment group: {label}\n\nComments:\n{blob}\n\nWrite a 5–7 sentence summary."},
        ],
        timeout=65,
    )
    return _enforce_single_paragraph((r.choices[0].message.content or "").strip(), max_sentences=7)

def _final_tldr_from_groups(sentiment: dict, pos_s: str, neu_s: str, neg_s: str) -> str:
    payload = (
        f"Distribution: positive={sentiment.get('pos',0)}%, "
        f"neutral={sentiment.get('neu',0)}%, negative={sentiment.get('neg',0)}%.\n\n"
        f"Positive summary:\n{pos_s or '[none]'}\n\n"
        f"Neutral summary:\n{neu_s or '[none]'}\n\n"
        f"Negative summary:\n{neg_s or '[none]'}"
    )
    r = _call_llm_with_retry(
        [
            {"role": "system", "content": _SYSTEM_FINAL},
            {"role": "user", "content": payload},
        ],
        timeout=75,
    )
    return _enforce_single_paragraph((r.choices[0].message.content or "").strip(), max_sentences=10)

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
@limiter.limit("20 per minute")
def index():
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

    sort = "top"  # always top
    summary = None
    pos_summary = neu_summary = neg_summary = ""
    data = {}
    error = None
    url = None

    if request.method == "POST":
        raw_url = (request.form.get("url") or "").strip()
        try:
            url = normalize_reddit_url(raw_url)
        except ValueError as ve:
            error = str(ve)
            return render_template("index.html", summary=None, error=error)

        cache_key = _cache_key(url, sort, OPENROUTER_MODEL)
        cached = _get_cache(cache_key)
        if cached:
            data, summary, pos_summary, neu_summary, neg_summary = cached
        else:
            try:
                data = fetch_submission_data(url, sort=sort, max_comments=None)
            except Exception as e:
                msg = str(e)
                if "timed out" in msg.lower() or "timeout" in msg.lower():
                    error = "Reddit is taking too long to respond. Please try again."
                else:
                    error = msg
                return render_template("index.html", summary=None, error=error)

            groups = data.get("groups") or {}
            pos_texts = [t for _, t in groups.get("pos", [])][:180]
            neu_texts = [t for _, t in groups.get("neu", [])][:180]
            neg_texts = [t for _, t in groups.get("neg", [])][:180]

            # Run the three group summaries in parallel
            with ThreadPoolExecutor(max_workers=3) as ex:
                f_pos = ex.submit(_summarize_group, "positive", pos_texts)
                f_neu = ex.submit(_summarize_group, "neutral", neu_texts)
                f_neg = ex.submit(_summarize_group, "negative", neg_texts)
                pos_summary = f_pos.result()
                neu_summary = f_neu.result()
                neg_summary = f_neg.result()

            summary = _final_tldr_from_groups(data.get("sentiment") or {}, pos_summary, neu_summary, neg_summary)
            _set_cache(cache_key, data, summary, pos_summary, neu_summary, neg_summary)

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
        "pos_summary": pos_summary,
        "neu_summary": neu_summary,
        "neg_summary": neg_summary,
        "current_sort": sort,
        "current_url": current_url,
        "sample_size": data.get("sample_size") if data else None,
        "effective_max": data.get("effective_max") if data else None,
    }
    return render_template("index.html", **payload)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5057, debug=True)
