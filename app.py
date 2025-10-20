import os
import re
import time
import json
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
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

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

def _call_llm_with_retry(messages, attempts: int = 3, timeout: int = 60, temperature: float = 0.2, top_p: float = 0.8):
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
                temperature=temperature,
                top_p=top_p,
                extra_body={"seed": 42},
                extra_headers=headers or None,
                timeout=timeout,
            )
            return r
        except Exception as e:
            last = e
            time.sleep(0.6 * (i + 1))
    raise last

# ====== NEW: Structured extractor prompts ======

_SYSTEM_GROUP = (
    "You summarize ONLY from the provided Reddit comments. "
    "No outside knowledge, no quotes, no usernames, no links. "
    "Return STRICT JSON that matches the schema—no extra text. "
    "Prefer specifics (pacing, motivation, choreography, etc.) over vibes. "
    "If evidence is weak/noisy, set low_confidence true."
)

_GROUP_USER_TMPL = """Task: Extract the strongest opinion units from this sentiment group: {label}

Rules:
- Cluster similar remarks and identify 3–6 recurring claims. Fewer if weak/noisy.
- Each claim must be concrete and grounded in the comments.
- Provide short paraphrase fragments that typify the claim (no quotes).
- Do NOT invent numbers or facts.

Output JSON schema:
{{
  "topic": "string",
  "sentiment": "{label}",
  "low_confidence": boolean,
  "claims": [
    {{
      "claim": "concise statement",
      "why_it_matters": "1 short reason",
      "evidence": ["3-10 word paraphrase fragments"],
      "prevalence_hint": "high|medium|low"
    }}
  ]
}}

Comments:
{blob}
"""

# ====== NEW: Final TL;DR prompt ======

_SYSTEM_FINAL = (
    "You write newsroom-quality TL;DRs. One paragraph. 110–140 words. "
    "No lists, no headers, no quotes, no hedging filler unless reflected in inputs. "
    "Prioritize claims with higher prevalence_hint and higher-confidence groups. "
    "Stay strictly inside the provided data."
)

_FINAL_USER_TMPL = """You are summarizing a Reddit thread about: {topic}

Structured inputs:
- positive_claims: {pos_json}
- neutral_claims: {neu_json}
- negative_claims: {neg_json}
- distribution_hint: positive={pos:.1f}%, neutral={neu:.1f}%, negative={neg:.1f}%

Write ONE paragraph that:
- Opens with a clear lede that reflects the dominant stance without naming percentages.
- Weaves 2–3 strongest supporting points with brief rationale.
- Acknowledges 1–2 key counterpoints.
- Concludes with a synthesis (what readers likely take away).
- Strict length: 110–140 words. Plain text only.
"""

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

# ====== NEW: extractor + renderer ======

def _extract_group_claims(label: str, texts: list[str]) -> dict:
    blob = " ".join(texts)[:4000]
    if not blob.strip():
        return {"topic": "", "sentiment": label, "low_confidence": True, "claims": []}
    r = _call_llm_with_retry(
        [
            {"role": "system", "content": _SYSTEM_GROUP},
            {"role": "user", "content": _GROUP_USER_TMPL.format(label=label, blob=blob)},
        ],
        timeout=65,
        temperature=0.2,   # precise extraction
        top_p=0.9          # allow some diversity for claim discovery
    )
    content = (r.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
        # minimal validation
        if not isinstance(data, dict) or "claims" not in data:
            raise ValueError("bad schema")
        data.setdefault("low_confidence", False)
        return data
    except Exception:
        # Fallback: create a single-claim structure from a plain summary
        fallback = _plain_summary_fallback(label, blob)
        return fallback

def _plain_summary_fallback(label: str, blob: str) -> dict:
    r = _call_llm_with_retry(
        [
            {"role": "system", "content": "Summarize in 5-6 concrete sentences. No quotes, no links."},
            {"role": "user", "content": blob[:3500]},
        ],
        timeout=45,
        temperature=0.3,
        top_p=0.6
    )
    summary = _enforce_single_paragraph((r.choices[0].message.content or "").strip(), max_sentences=6)
    return {
        "topic": "",
        "sentiment": label,
        "low_confidence": True,
        "claims": [{
            "claim": summary,
            "why_it_matters": "captures group gist",
            "evidence": [],
            "prevalence_hint": "medium"
        }]
    }

def _render_group_paragraph(struct: dict, max_claims: int = 5) -> str:
    """Turn structured claims into a tight paragraph (your prose, not the model's)."""
    claims = struct.get("claims", [])[:max_claims]
    if not claims:
        return ""
    # Sort by prevalence_hint
    order = {"high": 0, "medium": 1, "low": 2}
    claims.sort(key=lambda c: order.get(c.get("prevalence_hint","medium"), 1))
    parts = []
    for c in claims:
        claim = c.get("claim", "").strip()
        why = c.get("why_it_matters", "").strip()
        if claim and why:
            parts.append(f"{claim} ({why}).")
        elif claim:
            parts.append(claim if claim.endswith(".") else claim + ".")
    text = " ".join(parts)
    return _enforce_single_paragraph(text, max_sentences=7)

def _final_tldr_from_structs(sentiment: dict, pos_struct: dict, neu_struct: dict, neg_struct: dict) -> str:
    topic = (
        pos_struct.get("topic")
        or neu_struct.get("topic")
        or neg_struct.get("topic")
        or "the post"
    )
    payload = _FINAL_USER_TMPL.format(
        topic=topic,
        pos_json=json.dumps(pos_struct.get("claims", []), ensure_ascii=False),
        neu_json=json.dumps(neu_struct.get("claims", []), ensure_ascii=False),
        neg_json=json.dumps(neg_struct.get("claims", []), ensure_ascii=False),
        pos=float(sentiment.get("pos", 0)),
        neu=float(sentiment.get("neu", 0)),
        neg=float(sentiment.get("neg", 0)),
    )
    r = _call_llm_with_retry(
        [
            {"role": "system", "content": _SYSTEM_FINAL},
            {"role": "user", "content": payload},
        ],
        timeout=75,
        temperature=0.2,  # tighter prose
        top_p=0.3
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

            # Run the three group extractions in parallel
            with ThreadPoolExecutor(max_workers=3) as ex:
                f_pos = ex.submit(_extract_group_claims, "positive", pos_texts)
                f_neu = ex.submit(_extract_group_claims, "neutral", neu_texts)
                f_neg = ex.submit(_extract_group_claims, "negative", neg_texts)
                pos_struct = f_pos.result()
                neu_struct = f_neu.result()
                neg_struct = f_neg.result()

            # Render claims to group paragraphs for UI (keeps cache shape stable)
            pos_summary = _render_group_paragraph(pos_struct)
            neu_summary = _render_group_paragraph(neu_struct)
            neg_summary = _render_group_paragraph(neg_struct)

            # Final TL;DR using structured data
            summary = _final_tldr_from_structs(
                data.get("sentiment") or {},
                pos_struct, neu_struct, neg_struct
            )
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
