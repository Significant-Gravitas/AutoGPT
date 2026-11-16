import httpx
import openai
import pytest

from backend.util.llm.prompt_refusal import is_prompt_refused

# The body of OPEN-3666's Sentry event, as OpenAI sends it.
OPENAI_INVALID_PROMPT_BODY = {
    "error": {
        "message": (
            "Invalid prompt: your prompt was flagged as potentially violating "
            "our usage policy. Please try again with a different prompt: "
            "https://platform.openai.com/docs/guides/reasoning#advice-on-prompting"
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_prompt",
    }
}

# OpenRouter's ModerationErrorMetadata (openrouter.ai/docs/api-reference/errors).
_OPENROUTER_MODERATION_BODY = {
    "error": {
        "code": 403,
        "message": "openai/gpt-4o requires moderation. Your input was flagged.",
        "metadata": {
            "reasons": ["violence"],
            "flagged_input": "the user's prompt",
            "provider_name": "OpenAI",
            "model_slug": "openai/gpt-4o",
        },
    }
}


async def openai_sdk_error(status: int, body: dict) -> openai.APIStatusError:
    """The exception the OpenAI SDK itself raises for this HTTP response."""
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json=body))
    client = openai.AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    )
    with pytest.raises(openai.APIStatusError) as caught:
        await client.chat.completions.create(
            model="o3", messages=[{"role": "user", "content": "the user's prompt"}]
        )
    return caught.value


@pytest.mark.parametrize(
    "status, body, refused",
    [
        (400, OPENAI_INVALID_PROMPT_BODY, True),
        (
            400,
            {"error": {"code": "content_policy_violation", "message": "Rejected"}},
            True,
        ),
        (403, _OPENROUTER_MODERATION_BODY, True),
        (
            400,
            {
                "error": {
                    "code": "context_length_exceeded",
                    "message": "flagged as potentially violating our usage policy",
                }
            },
            False,
        ),
        (403, {"error": {"code": 403, "message": "Key lacks access"}}, False),
    ],
    ids=[
        "openai-invalid-prompt",
        "openai-content-policy",
        "openrouter-moderation",
        "other-400-quoting-the-wording",
        "openrouter-403-not-moderation",
    ],
)
async def test_only_a_content_refusal_is_classified(
    status: int, body: dict, refused: bool
):
    assert is_prompt_refused(await openai_sdk_error(status, body)) is refused
