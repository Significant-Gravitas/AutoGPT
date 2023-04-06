import openai
from config import Config
cfg = Config()

openai.api_key = cfg.openai_api_key

# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message["content"]


@retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5))
def create_chat_completion_with_retry(messages):
    return create_chat_completion(
        model=cfg.fast_llm_model,
        messages=messages,
        max_tokens=300,
    )
