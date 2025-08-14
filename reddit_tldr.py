import os
import re
import math
from typing import List, Tuple
from pathlib import Path

import praw
from praw.models import MoreComments
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the .env that is next to this file (works regardless of CWD)
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")  # blank allowed for Installed App
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "reddit-tldr-bot")

if not (REDDIT_CLIENT_ID and REDDIT_USER_AGENT):
    raise RuntimeError("Missing Reddit creds in .env (REDDIT_CLIENT_ID & REDDIT_USER_AGENT).")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET if REDDIT_CLIENT_SECRET is not None else "",
    user_agent=REDDIT_USER_AGENT,
)

_analyzer = SentimentIntensityAnalyzer()

# ---- public helpers used by app.py ----
def verify_reddit_credentials() -> None:
    try:
        _ = reddit.read_only
        list(reddit.subreddit("python").hot(limit=1))
    except Exception as e:
        hint = (
            "Reddit auth failed.\n"
            "Likely causes:\n"
            "- Wrong REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET\n"
            "- App type mismatch:\n"
            "    * If your app is **Installed App**, set REDDIT_CLIENT_SECRET= (blank)\n"
            "    * If it's **Script/Web App**, it must have a real secret\n"
            "- Trailing spaces/newlines in .env values\n"
            "- USER_AGENT missing or too generic\n"
            "Check https://www.reddit.com/prefs/apps and update your .env."
        )
        raise RuntimeError(f"{e}\n\n{hint}") from e


def _sentiment_and_groups(comments: List[str]) -> tuple[dict, dict, int]:
    """
    VADER per comment; percentages softly weighted by upvotes if present in [score=..] tags.
    Returns (percentages, groups, sample_size).
    """
    bins = {"pos": 0.0, "neu": 0.0, "neg": 0.0}
    groups: dict[str, list[tuple[float, str]]] = {"pos": [], "neu": [], "neg": []}
    n = 0
    score_re = re.compile(r"\[score=(-?\d+)")

    for body in comments:
        try:
            s = _analyzer.polarity_scores(body or "")
            comp = s.get("compound", 0.0)

            # weight by score (1.0–4.0 roughly)
            w = 1.0
            m = score_re.search(body)
            if m:
                try:
                    sc = max(0, int(m.group(1)))
                    w = 1.0 + min(3.0, math.log10(1.0 + sc))
                except Exception:
                    pass

            if comp >= 0.05:
                bins["pos"] += w; groups["pos"].append((comp, body))
            elif comp <= -0.05:
                bins["neg"] += w; groups["neg"].append((comp, body))
            else:
                bins["neu"] += w; groups["neu"].append((comp, body))
            n += 1
        except Exception:
            continue

    total_w = (bins["pos"] + bins["neu"] + bins["neg"]) or 1.0
    pct = {k: round(v / total_w * 100.0, 1) for k, v in bins.items()}

    # most informative first; trim to keep prompts tight
    groups["pos"].sort(key=lambda t: t[0], reverse=True)
    groups["neu"].sort(key=lambda t: abs(t[0]), reverse=True)
    groups["neg"].sort(key=lambda t: abs(t[0]), reverse=True)
    for k in ("pos", "neu", "neg"):
        groups[k] = groups[k][:300]

    return pct, groups, n


def fetch_submission_data(url: str, *, sort: str = "top", max_comments: int | None = None) -> dict:
    """
    Auto-sizes when max_comments is None:
      - cap at 300
      - ensure at least 50
      - if thread is large, lightly expand and cap replies per top-level
      - avoid full-tree flattening on big threads
    """
    if not url or ("reddit.com" not in url and "redd.it" not in url):
        raise ValueError("Please provide a valid Reddit post URL.")

    submission = reddit.submission(url=url)
    _ = submission.title  # force fetch

    total = int(getattr(submission, "num_comments", 0) or 0)
    if max_comments is None:
        if total <= 60:
            eff = max(50, total)
        elif total <= 150:
            eff = 150
        else:
            eff = 300
    else:
        eff = max(50, min(1200, int(max_comments)))

    _apply_sort(submission, sort)

    comments: List[str] = []
    kept_for_dedup: List[str] = []

    def add_text(txt: str):
        if _filter_comment(txt):
            if not _too_similar(txt, kept_for_dedup):
                kept_for_dedup.append(txt)
                comments.append(txt)

    if total <= 80:
        # Small/easy: expand fully once and finish.
        submission.comment_limit = min(400, max(60, total + 40))
        submission.comments.replace_more(limit=0)  # fully resolve (small thread)
        for node in submission.comments.list():
            if isinstance(node, MoreComments):
                continue
            body = getattr(node, "body", "") or ""
            if body:
                add_text(_with_meta(node))
                if len(comments) >= eff:
                    break
    else:
        # Big threads: keep network light. Do a shallow expansion and cap replies.
        submission.comment_limit = min(800, eff + 120)
        submission.comments.replace_more(limit=2, threshold=32)  # light expand only

        top_level = list(submission.comments)
        reply_cap = 3 if total >= 150 else 2

        for c in top_level:
            if len(comments) >= eff:
                break
            try:
                add_text(_with_meta(c))
            except Exception:
                continue

            taken = 0
            # Use whatever replies are already loaded; don't force deeper expansion.
            try:
                for r in (c.replies or [])[:reply_cap]:
                    if len(comments) >= eff:
                        break
                    add_text(_with_meta(r))
                    taken += 1
            except Exception:
                pass

        # If under budget, do a shallow sweep of already-loaded subtree (no extra replace_more)
        if len(comments) < eff:
            for c in top_level:
                if len(comments) >= eff:
                    break
                try:
                    for r in (c.replies or []):
                        if len(comments) >= eff:
                            break
                        add_text(_with_meta(r))
                except Exception:
                    continue

    if not comments:
        raise RuntimeError("No usable comments were found for that post.")

    paragraph = _make_paragraph([_strip_meta_for_para(t) for t in comments], sentence_limit=70)
    image_url, video_url = _extract_media(submission)

    sentiment, groups, sample_n = _sentiment_and_groups(comments)

    return {
        "title": submission.title or "",
        "subreddit": str(submission.subreddit),
        "selftext": submission.selftext or "",
        "paragraph": paragraph,
        "image_url": image_url,
        "video_url": video_url,
        "sentiment": sentiment,
        "groups": groups,
        "effective_max": eff,
        "sample_size": sample_n,
    }

# ---- internal helpers ----
def _apply_sort(submission, sort: str):
    sort = (sort or "top").lower()
    if sort == "new": submission.comment_sort = "new"
    elif sort == "controversial": submission.comment_sort = "controversial"
    elif sort == "best": submission.comment_sort = "best"
    elif sort == "hot": submission.comment_sort = "hot"
    elif sort == "old": submission.comment_sort = "old"
    elif sort in ("q&a", "qa"): submission.comment_sort = "qa"
    else: submission.comment_sort = "top"

def _with_meta(c) -> str:
    body = (getattr(c, "body", "") or "").strip()
    if not body:
        return ""
    score = getattr(c, "score", 0) or 0
    replies = len(getattr(c, "replies", []) or [])
    author = getattr(getattr(c, "author", None), "name", "") or ""
    return f"[score={score} replies={replies} author=u/{author}] {body}"

def _strip_meta_for_para(s: str) -> str:
    return re.sub(r"^\s*\[[^\]]+\]\s*", "", s or "").strip()

def _extract_media(submission) -> Tuple[str | None, str | None]:
    image_url, video_url = None, None
    post_hint = getattr(submission, "post_hint", "") or ""
    try:
        if post_hint in ("image", "hosted:image"):
            image_url = submission.preview["images"][0]["source"]["url"]
        elif post_hint in ("hosted:video", "rich:video"):
            media = submission.media or {}
            rv = media.get("reddit_video") if isinstance(media, dict) else None
            if rv and "fallback_url" in rv:
                video_url = rv["fallback_url"]
    except Exception:
        pass
    return image_url, video_url

def _filter_comment(body: str) -> bool:
    if not body or len(body.strip()) < 3:
        return False
    t = body.lower().strip()
    if t in ("[deleted]", "[removed]"):
        return False
    if "http://" in t or "https://" in t or "www." in t:
        return False
    return True

def _too_similar(text: str, kept: List[str], threshold: float = 0.85) -> bool:
    tokens_a = set(re.findall(r"[a-zA-Z0-9']+", text.lower()))
    if not tokens_a:
        return False
    for k in kept[-60:]:
        tokens_b = set(re.findall(r"[a-zA-Z0-9']+", k.lower()))
        if not tokens_b:
            continue
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        if union and (inter / union) >= threshold:
            return True
    return False

def _make_paragraph(comments: List[str], sentence_limit: int = 70) -> str:
    text = " ".join(comments).replace("\n", " ").strip('"')
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = [s.strip() if s.endswith(('.', '!', '?')) else (s.strip() + ".") for s in parts if s.strip()]
    return "".join((s + " ") for s in sentences[:sentence_limit])[:20000].strip()
