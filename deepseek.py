from openai import OpenAI
import reddit_tldr

para = reddit_tldr.paragraph
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_body={},
  model="deepseek/deepseek-r1-0528:free",
  messages=[
    {
      "role": "user",
      "content": ("This text is from a reddit post, specifically the comments under it. Summarize this text and make a TLDR paragraph first, and then numbered bullet points with important points and perspectives from comments and replies under that and DO NOT give anything further. Please try to make it as readable and clean as possible." + para)
    }
  ]
)
print(completion.choices[0].message.content)