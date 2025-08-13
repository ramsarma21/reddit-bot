import os
import re
from typing import List, Tuple
from pathlib import Path

import praw
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

# If your Reddit app is 'Installed App', client_secret must be blank; if 'Script/Web App', it must be set.
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET if REDDIT_CLIENT_SECRET is not None else "",
    user_agent=REDDIT_USER_AGENT,
)

_analyzer = SentimentIntensityAnalyzer()

# ---- public helpers used by app.py ----
def verify_reddit_credentials() -> None:
    """Raise a clear error if Reddit credentials are wrong or app type is mismatched."""
    try:
        _ = reddit.read_only
        # low-cost request to ensure creds/app type are valid
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


def fetch_submission_data(url: str, *, sort: str = "top", max_comments: int = 500) -> dict:
    """
    Return:
      {
        title, subreddit, selftext, paragraph, image_url, video_url, sentiment
      }
    """
    if not url or "reddit.com" not in url:
        raise ValueError("Please provide a valid Reddit post URL.")

    submission = reddit.submission(url=url)
    _ = submission.title  # force fetch
    _apply_sort(submission, sort)
    submission.comments.replace_more(limit=0)

    comments: List[str] = []
    for c in submission.comments.list():
        try:
            if hasattr(c, "body") and _filter_comment(c.body, getattr(c, "author", None)):
                comments.append(c.body.strip())
                if len(comments) >= max_comments:
                    break
        except Exception:
            continue

    if not comments:
        raise RuntimeError("No usable comments were found for that post.")

    paragraph = _make_paragraph(comments, sentence_limit=70)
    sentiment = _sentiment_split(comments)
    image_url, video_url = _extract_media(submission)

    return {
        "title": submission.title or "",
        "subreddit": str(submission.subreddit),
        "selftext": submission.selftext or "",
        "paragraph": paragraph,
        "image_url": image_url,
        "video_url": video_url,
        "sentiment": sentiment,
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


def _filter_comment(body: str, author) -> bool:
    """
    Keep deleted/removed/very-short/bot/link-only/mod comments out.
    Profanity is allowed here (LLM prompt sanitizes final output).
    """
    if not body:
        return False
    text = body.strip()
    if not text or len(text) <= 10:
        return False
    low = text.lower()
    if low in ("[deleted]", "[removed]"):
        return False
    if "i am a bot" in low:
        return False
    if ("http://" in low or "https://" in low) and len(low.replace("http://", "").replace("https://", "")) < 40:
        return False
    try:
        if author is not None and getattr(author, "is_mod", False):
            return False
    except Exception:
        pass
    return True


def _make_paragraph(comments: List[str], sentence_limit: int = 70) -> str:
    """Original style: split by '.', restore '. ', take first N sentences; cap for token safety."""
    text = " ".join(comments).replace("\n", " ").strip('"')
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    sentences = [s + ". " for s in sentences]
    return "".join(sentences[:sentence_limit])[:20000]


def _sentiment_split(comments: List[str]) -> dict:
    """Return percentage split of positive/neutral/negative using VADER."""
    bins = {"pos": 0, "neu": 0, "neg": 0}
    total = 0
    for body in comments:
        try:
            s = _analyzer.polarity_scores(body or "")
            comp = s.get("compound", 0.0)
            if comp >= 0.05:
                bins["pos"] += 1
            elif comp <= -0.05:
                bins["neg"] += 1
            else:
                bins["neu"] += 1
            total += 1
        except Exception:
            continue
    if total == 0:
        return {"pos": 0.0, "neu": 0.0, "neg": 0.0}
    return {k: round(v / total * 100.0, 1) for k, v in bins.items()}
