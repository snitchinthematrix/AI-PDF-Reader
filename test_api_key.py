from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);