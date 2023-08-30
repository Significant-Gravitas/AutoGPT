import openai

openai.api_base = "http://localhost:4891/v1"

openai.api_key = "not needed for a local LLM"


model = "ggml-llama-2-13b-chat.ggmlv3.q4_0.bin"
prompt = "Who is Michael Jordan?"
response = openai.Completion.create(
    model=model,
    prompt=prompt,
    max_tokens=50,
    temperature=0.28,
    top_p=0.95,
    n=1,
    echo=True,
    stream=False,
)
assert len(response["choices"][0]["text"]) > len(prompt)
print(f"Model: {response['model']}")
print(f"Usage: {response['usage']}")
print(f"Answer: {response['choices'][0]['text']}")
