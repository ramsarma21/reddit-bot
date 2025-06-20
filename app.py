from flask import Flask, request, render_template
from reddit_tldr import fetch_submission_data
from openai import OpenAI

app = Flask(__name__)

# Hugging Face / OpenRouter client setup
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

def summarize_with_openrouter(paragraph):
    prompt = (
        "This text is from a Reddit post, specifically the comments under it. "
        "Summarize this text and make a TLDR paragraph first, and then numbered bullet points "
        "with important points and perspectives from comments and replies. "
        "DO NOT give anything further. Make it clean, readable, and concise. "
        "Try to make it 5-6 bullet points, they can be a bit longer, but make sure to cover important parts. "
         "Also do not add toxic and insulting comments without value to the summaries. "
         " \n\n"
        + paragraph
    )

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528:free",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    data = {}
    error = None

    if request.method == "POST":
        url = request.form.get("url")
        try:
            data = fetch_submission_data(url)
            summary = summarize_with_openrouter(data["paragraph"])
        except Exception as e:
            error = str(e)

    return render_template("index.html", summary=summary, error=error, **data)

if __name__ == "__main__":
    app.run(debug=True)
