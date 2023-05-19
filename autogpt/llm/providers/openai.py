from autogpt.llm.base import ChatModelInfo, EmbeddingModelInfo

# 15% less of max tokens
SOFT_TOKEN_LIMIT = 0.85

OPEN_AI_CHAT_MODELS = {
    "gpt-3.5-turbo": ChatModelInfo(
        name="gpt-3.5-turbo",
        prompt_token_cost=0.002,
        completion_token_cost=0.002,
        max_tokens=4096,
        soft_token_limit=int(4096 * SOFT_TOKEN_LIMIT),
    ),
    "gpt-4": ChatModelInfo(
        name="gpt-4",
        prompt_token_cost=0.03,
        completion_token_cost=0.06,
        max_tokens=8192,
        soft_token_limit=int(8192 * SOFT_TOKEN_LIMIT),
    ),
    "gpt-4-32k": ChatModelInfo(
        name="gpt-4-32k",
        prompt_token_cost=0.06,
        completion_token_cost=0.12,
        max_tokens=32768,
        soft_token_limit=int(32768 * SOFT_TOKEN_LIMIT),
    ),
}

OPEN_AI_EMBEDDING_MODELS = {
    "text-embedding-ada-002": EmbeddingModelInfo(
        name="text-embedding-ada-002",
        prompt_token_cost=0.0004,
        completion_token_cost=0.0,
        max_tokens=8191,
        soft_token_limit=int(8191 * SOFT_TOKEN_LIMIT),
        embedding_dimensions=1536,
    ),
}

OPEN_AI_MODELS = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}
