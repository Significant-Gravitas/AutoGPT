DEFAULT_LLM_MODEL = "gpt-6-luna"


def is_luna_model(model: str) -> bool:
    # An OpenRouter variant (``gpt-6-luna:nitro``) is the same model.
    return model.removeprefix("openai/").split(":", 1)[0] == DEFAULT_LLM_MODEL
