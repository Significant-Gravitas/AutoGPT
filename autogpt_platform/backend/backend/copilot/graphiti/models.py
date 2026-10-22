DEFAULT_LLM_MODEL = "gpt-6-luna"


def is_luna_model(model: str) -> bool:
    return model.removeprefix("openai/") == DEFAULT_LLM_MODEL
