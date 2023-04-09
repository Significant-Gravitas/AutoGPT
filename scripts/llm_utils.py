import openai
from config import Config
cfg = Config()

openai.api_key = cfg.openai_api_key

def get_openai_deployment_id(model: str):
    match model:
        case cfg.fast_llm_model:
            return cfg.fast_llm_model_deployment_id
        case cfg.smart_llm_model:
            return cfg.smart_llm_model_deployment_id
        # TODO: obtain/create deployment id by model from azure CLI
        # default to GPT3.5 Only Mode if model selection is ambiguous
        case default: 
            return cfg.fast_llm_model_deployment_id


# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=get_openai_deployment_id(model),
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return response.choices[0].message["content"]
