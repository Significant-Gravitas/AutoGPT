from autogpt.llm.base import ModelInfo, ModelType

OPEN_AI_CHAT_MODELS = {
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        model_type=ModelType.chat,
        prompt_token_cost=0.002,
        completion_token_cost=0.002,
        max_tokens=4096,
    ),
    "gpt-4": ModelInfo(
        name="gpt-4",
        model_type=ModelType.chat,
        prompt_token_cost=0.03,
        completion_token_cost=0.06,
        max_tokens=8192,
    ),
    "gpt-4-32k": ModelInfo(
        name="gpt-4-32k",
        model_type=ModelType.chat,
        prompt_token_cost=0.06,
        completion_token_cost=0.12,
        max_tokens=32768,
    ),
}

OPEN_AI_EMBEDDING_MODELS = {
    "text-embedding-ada-002": ModelInfo(
        name="text-embedding-ada-002",
        model_type=ModelType.embedding,
        prompt_token_cost=0.0004,
        completion_token_cost=0.0,
        max_tokens=8191,
    ),
}

OPEN_AI_MODELS = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}
