import openai
openai.api_key = "YOUR_API_KEY"

def convert_to_ai(input_text):
    prompt = (f"Convert {input_text} to AGENTGPT-AI4DBOT AI")

    response = openai.Completion.create(
      engine="davinci-codex",
      prompt=prompt,
      temperature=0.9,
      max_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    message = response.choices[0].text
    return message
