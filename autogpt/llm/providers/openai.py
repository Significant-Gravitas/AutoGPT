from autogpt.llm.base import ChatModelInfo, EmbeddingModelInfo, TextModelInfo

# 15% less of max tokens
SOFT_TOKEN_LIMIT = 0.85

OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name="gpt-3.5-turbo",
            prompt_token_cost=0.002,
            completion_token_cost=0.002,
            max_tokens=4096,
            soft_token_limit=4096 * SOFT_TOKEN_LIMIT,
        ),
        ChatModelInfo(
            name="gpt-3.5-turbo-0301",
            prompt_token_cost=0.002,
            completion_token_cost=0.002,
            max_tokens=4096,
            soft_token_limit=4096 * SOFT_TOKEN_LIMIT,
        ),
        ChatModelInfo(
            name="gpt-4",
            prompt_token_cost=0.03,
            completion_token_cost=0.06,
            max_tokens=8192,
            soft_token_limit=8192 * SOFT_TOKEN_LIMIT,
        ),
        ChatModelInfo(
            name="gpt-4-0314",
            prompt_token_cost=0.03,
            completion_token_cost=0.06,
            max_tokens=8192,
            soft_token_limit=8192 * SOFT_TOKEN_LIMIT,
        ),
        ChatModelInfo(
            name="gpt-4-32k",
            prompt_token_cost=0.06,
            completion_token_cost=0.12,
            max_tokens=32768,
            soft_token_limit=32768 * SOFT_TOKEN_LIMIT,
        ),
        ChatModelInfo(
            name="gpt-4-32k-0314",
            prompt_token_cost=0.06,
            completion_token_cost=0.12,
            max_tokens=32768,
            soft_token_limit=32768 * SOFT_TOKEN_LIMIT,
        ),
    ]
}

OPEN_AI_TEXT_MODELS = {
    info.name: info
    for info in [
        TextModelInfo(
            name="text-davinci-003",
            prompt_token_cost=0.02,
            completion_token_cost=0.02,
            max_tokens=4097,
            soft_token_limit=4097 * SOFT_TOKEN_LIMIT,
        ),
    ]
}

OPEN_AI_EMBEDDING_MODELS = {
    info.name: info
    for info in [
        EmbeddingModelInfo(
            name="text-embedding-ada-002",
            prompt_token_cost=0.0004,
            completion_token_cost=0.0,
            max_tokens=8191,
            soft_token_limit=8191 * SOFT_TOKEN_LIMIT,
            embedding_dimensions=1536,
        ),
    ]
}

OPEN_AI_MODELS: dict[str, ChatModelInfo | EmbeddingModelInfo | TextModelInfo] = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_TEXT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}
