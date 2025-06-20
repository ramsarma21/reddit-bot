import praw
import pprint

# from gensim.summarization import summarize

reddit = praw.Reddit(
    client_id="JwTi5xVBbiDR_nc09PYU6Q",
    client_secret="YwJccYDyxb-N4WLuUTsgPPZz-UheJQ",
    user_agent="reddit-tldr-bot by u/Special_Big740"
)


def fetch_submission_data(url):
    banned_words = ["nigga", "nigger", "faggot", "pedophile", "pdfile", "retard"]
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)
    comments = [
    c.body.strip()
    for c in submission.comments.list()
    if hasattr(c, "body")
    and c.body.lower() not in ("[deleted]", "[removed]") #deleted comments
    and len(c.body.strip()) > 10 #short comments
    and "i am a bot" not in c.body.strip().lower() #bot comments
    and "http" not in c.body.strip().lower()  #link only comments
    and (c.author is None or "bot" not in c.author.name.lower())
    and not any(banned_word in c.body.lower() for banned_word in banned_words)
]
    
    # ---- format comment paragraph ----
    text = " ".join(comments).replace("\n", "").strip('"')
    sentences = text.split(".")
    sentences = [s.strip() + ". " for s in sentences if s.strip()]
    paragraph = "".join(sentences[:70])

    # ---- extract image or video link if available ----
    image_url = None
    video_url = None

    if hasattr(submission, "post_hint"):
        if submission.post_hint == "image":
            image_url = submission.preview['images'][0]['source']['url']
        elif submission.post_hint == "hosted:video":
            video_url = submission.media['reddit_video']['fallback_url']

    return {
        "title": submission.title,
        "subreddit": str(submission.subreddit),
        "selftext": submission.selftext,
        "paragraph": paragraph,
        "image_url": image_url,
        "video_url": video_url,
    }

