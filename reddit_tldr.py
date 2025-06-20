import praw
import pprint

# from gensim.summarization import summarize

reddit = praw.Reddit(
    client_id="JwTi5xVBbiDR_nc09PYU6Q",
    client_secret="YwJccYDyxb-N4WLuUTsgPPZz-UheJQ",
    user_agent="reddit-tldr-bot by u/Special_Big740"
)

url = input("Enter your url: ")
submission = reddit.submission(url=url)
# print(submission.title)  # to make it non-lazy  **CHECK ALL FUNCTIONS**
# pprint.pprint(vars(submission))

if submission.post_hint == "hosted:image":  #check if post has image and store link
    image_link = (submission.preview['images'][0]['source']['url'])
if submission.post_hint == "hosted:video":  #check if post has video and store link
    video_link = submission.media['reddit_video']['fallback_url']
print("Post title: " + submission.title)
if (submission.selftext != ""):
    print("Text content: " + submission.selftext)
print("Subreddit of origin: " + str(submission.subreddit))

list_comments = [] #create comment list

submission.comments.replace_more(limit=0)  #add comments to comment list
for comment in submission.comments.list():
    list_comments.append(comment.body)

#**FORMATTING**
a=list(map(str,list_comments)) 
b=" ".join(a)
b=b.replace(str("\n"),"")
b=b.strip('"')
b=b.split(".")  #split each index into sentences
final_list=list(map(str,b))

for i in range(len(final_list)): #add periods to each sentence
    final_list[i] += ". "

paragraph = "" 
for i in final_list[:70]: #add first 60 sentences
    paragraph += i + ""         

# print(paragraph)                             


