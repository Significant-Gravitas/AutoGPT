import openai
from config import Config
cfg = Config()

openai.api_key = cfg.openai_api_key

# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=None, 
                           presence_penalty=cfg.presence_penalty, frequency_penalty=cfg.frequency_penalty, 
                           end_user=cfg.end_user)->str:
    """Create a chat completion using the OpenAI API"""
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.get_azure_deployment_id_for_model(model),
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            user=end_user
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            user=end_user
        )

    return response.choices[0].message["content"]
