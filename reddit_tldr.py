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
    if not url or ("reddit.com" not in url and "redd.it" not in url):
        raise ValueError("Please provide a valid Reddit post URL.")

    submission = reddit.submission(url=url)
    _ = submission.title  # force fetch
    _apply_sort(submission, sort)
    submission.comments.replace_more(limit=0)

    comments: List[str] = []
    kept: List[str] = []
    for c in submission.comments.list():
        try:
            if hasattr(c, "body") and _filter_comment(c.body, getattr(c, "author", None)):
                text = c.body.strip()
                # de-dup near-duplicates to cut boilerplate ("me too", copypasta)
                if not _too_similar(text, kept):
                    kept.append(text)
                    comments.append(text)
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


# -------- Comment Filtering 2.0 --------
_BOT_MARKERS = (
    "i am a bot",
    "automoderator",
    "this action was performed automatically",
)
_URL_RE = re.compile(r"(https?://\S+)")
_MIN_LEN = 12
# if comment is mostly URL gibberish, drop it
def _mostly_linky(text: str) -> bool:
    nonspace = len(re.sub(r"\s+", "", text))
    links = _URL_RE.findall(text)
    link_len = sum(len(L) for L in links)
    return bool(links) and (link_len / max(1, nonspace) > 0.6)

def _filter_comment(body: str, author) -> bool:
    """Return True if this comment should be kept, False if it should be excluded."""
    if not body or len(body.strip()) < 3:
        return False

    lower_body = body.lower().strip()

    # Skip deleted/removed/moderator/bot comments
    if lower_body in ("[deleted]", "[removed]"):
        return False
    if author and hasattr(author, "name") and author.name and "bot" in author.name.lower():
        return False
    if "moderator" in lower_body:
        return False

    # Skip if comment contains any URL
    if "http://" in lower_body or "https://" in lower_body or "www." in lower_body:
        return False

    # Skip if it's just a link or ad
    if len(lower_body.split()) <= 3 and ("http" in lower_body or "www." in lower_body):
        return False

    return True


    # Author checks
    try:
        if author is not None:
            name = str(getattr(author, "name", "") or "").lower()
            if name == "automoderator":
                return False
            if getattr(author, "is_mod", False):
                return False
    except Exception:
        pass

    # Drop URL-only or mostly-URL blurbs
    if _mostly_linky(text):
        return False

    # Drop trivial link-only lines
    if _URL_RE.fullmatch(text.strip()):
        return False

    return True

# Jaccard similarity de-dup to cut near-duplicates
def _too_similar(text: str, kept: List[str], threshold: float = 0.85) -> bool:
    import re
    tokens_a = set(re.findall(r"[a-zA-Z0-9']+", text.lower()))
    if not tokens_a:
        return False
    for k in kept[-60:]:  # only compare to recent to keep it cheap
        tokens_b = set(re.findall(r"[a-zA-Z0-9']+", k.lower()))
        if not tokens_b:
            continue
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        if union and (inter / union) >= threshold:
            return True
    return False

def _make_paragraph(comments: List[str], sentence_limit: int = 70) -> str:
    """
    Normalize whitespace, split on sentence-ish boundaries, restore punctuation,
    cap by sentence and length to keep LLM prompt efficient.
    """
    text = " ".join(comments).replace("\n", " ").strip('"')
    text = re.sub(r"\s+", " ", text).strip()

    # simple sentence split that keeps punctuation
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = [s.strip() if s.endswith(('.', '!', '?')) else (s.strip() + ".") for s in parts if s.strip()]
    return "".join((s + " ") for s in sentences[:sentence_limit])[:20000].strip()

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
