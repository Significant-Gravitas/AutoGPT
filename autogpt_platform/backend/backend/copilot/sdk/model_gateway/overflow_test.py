import re

import pytest

from backend.copilot.sdk.model_gateway.errors import parse_upstream_error
from backend.copilot.sdk.model_gateway.overflow import (
    ContextOverflow,
    detect_overflow,
    prompt_too_long,
    upstream_failure,
)

# The one wording CLI 2.1.284 compacts on.
_CLI_WORDING = re.compile(r"^prompt is too long: (\d+) tokens > (\d+) maximum$")

# Every body below is what the named server sends when the prompt does not
# fit; the figures are the ones it reports, or None where it reports none.
_UPSTREAM_OVERFLOWS = [
    (
        "llama.cpp",
        b'{"error":{"code":400,"message":"request (40000 tokens) exceeds the'
        b' available context size (32768 tokens), try increasing it","type":'
        b'"exceed_context_size_error","n_prompt_tokens":40000,"n_ctx":32768}}',
        (40000, 32768),
    ),
    (
        "llama.cpp, older wording",
        b'{"error":{"code":400,"message":"the request exceeds the available'
        b" context size, try increasing the context size or enable context"
        b' shift","type":"invalid_request_error"}}',
        (None, None),
    ),
    (
        "vLLM",
        b'{"object":"error","message":"This model\'s maximum context length is'
        b" 32768 tokens. However, your request has 40000 input tokens. Please"
        b' reduce the length of the input messages.","type":"BadRequestError",'
        b'"param":null,"code":400}',
        (40000, 32768),
    ),
    (
        "vLLM, prompt plus max_tokens",
        b'{"error":{"message":"This model\'s maximum context length is 32768'
        b" tokens and your request has 31000 input tokens (2048 > 32768 -"
        b' 31000).","type":"BadRequestError","code":400}}',
        (31000, 32768),
    ),
    (
        "OpenAI",
        b'{"error":{"message":"This model\'s maximum context length is 128000'
        b" tokens. However, your messages resulted in 130000 tokens. Please"
        b' reduce the length of the messages.","type":"invalid_request_error",'
        b'"param":"messages","code":"context_length_exceeded"}}',
        (130000, 128000),
    ),
    (
        "Ollama",
        b'{"error":"prompt too long; exceeded max context length by 7232 tokens"}',
        (None, None),
    ),
    (
        "LM Studio",
        b'{"error":"Trying to keep the first 40000 tokens when context the'
        b" overflows. However, the model is loaded with context length of only"
        b" 32768 tokens, which is not enough. Try to load the model with a"
        b' larger context length, or provide a shorter input"}',
        (40000, 32768),
    ),
    (
        "SGLang",
        b'{"object":"error","message":"The input (40000 tokens) is longer than'
        b' the model\'s context length (32768 tokens).","code":400}',
        (40000, 32768),
    ),
    (
        "TGI",
        b'{"error":"Input validation error: `inputs` tokens + `max_new_tokens`'
        b" must be <= 32768. Given: 40000 `inputs` tokens and 1000"
        b' `max_new_tokens`","error_type":"validation"}',
        (40000, 32768),
    ),
    (
        "Mistral",
        b'{"object":"error","message":"Prompt contains 40000 tokens and 0 draft'
        b' tokens, too large for model with 32768 maximum context length",'
        b'"type":"invalid_request_message_error","code":"3051"}',
        (40000, 32768),
    ),
    (
        "Gemini's OpenAI layer",
        b'[{"error":{"code":400,"message":"The input token count (40000)'
        b' exceeds the maximum number of tokens allowed (32768).",'
        b'"status":"INVALID_ARGUMENT"}}]',
        (40000, 32768),
    ),
    (
        "LiteLLM",
        b'{"error":{"message":"litellm.ContextWindowExceededError: litellm.'
        b"BadRequestError: ContextWindowExceededError: OpenAIException - request"
        b" (40000 tokens) exceeds the available context size (32768 tokens),"
        b' try increasing it","type":"invalid_request_error","code":"400"}}',
        (40000, 32768),
    ),
    (
        "Codex backend",
        b'{"error":{"code":"context_length_exceeded","message":"Your input'
        b" exceeds the context window of this model. Please adjust your input"
        b' and try again.","type":"invalid_request_error"}}',
        (None, None),
    ),
    (
        "Anthropic itself",
        b'{"type":"error","error":{"type":"invalid_request_error","message":'
        b'"prompt is too long: 40000 tokens > 32768 maximum"}}',
        (40000, 32768),
    ),
]


@pytest.mark.parametrize(
    "server, body, figures",
    _UPSTREAM_OVERFLOWS,
    ids=[case[0] for case in _UPSTREAM_OVERFLOWS],
)
def test_every_upstream_overflow_becomes_the_wording_the_cli_compacts_on(
    server, body, figures
):
    error = upstream_failure(
        parse_upstream_error(body),
        status=400,
        estimate=21_000,
        window=20_000,
        secrets=(),
    )

    assert error.status == 400
    assert error.error_type == "invalid_request_error"
    match = _CLI_WORDING.match(error.message)
    assert match, error.message
    prompt, limit = int(match.group(1)), int(match.group(2))
    assert prompt > limit
    expected_prompt, expected_limit = figures
    if expected_limit is not None:
        assert (prompt, limit) == (
            max(expected_prompt, expected_limit + 1),
            expected_limit,
        )


def test_the_upstream_figures_win_over_the_estimate_and_the_window():
    overflow = ContextOverflow(prompt_tokens=40_000, limit=32_768)
    error = prompt_too_long(overflow, estimate=10_000, window=100_000)
    assert error.message == "prompt is too long: 40000 tokens > 32768 maximum"


def test_an_excess_alone_is_added_to_the_window():
    overflow = detect_overflow(
        parse_upstream_error(
            b'{"error":"prompt too long; exceeded max context'
            b' length by 500 tokens"}'
        )
    )
    assert overflow == ContextOverflow(excess=500)
    error = prompt_too_long(overflow, estimate=1_000, window=32_768)
    assert error.message == "prompt is too long: 33268 tokens > 32768 maximum"


def test_with_no_figures_the_estimate_and_window_fill_in():
    error = prompt_too_long(ContextOverflow(), estimate=280_000, window=272_000)
    assert error.message == "prompt is too long: 280000 tokens > 272000 maximum"


def test_with_no_window_the_limit_sits_just_under_the_estimate():
    error = prompt_too_long(ContextOverflow(), estimate=50_000, window=None)
    assert error.message == "prompt is too long: 50000 tokens > 49999 maximum"


def test_an_estimate_under_the_window_still_states_a_prompt_over_it():
    error = prompt_too_long(ContextOverflow(), estimate=1_000, window=32_768)
    assert error.message == "prompt is too long: 32769 tokens > 32768 maximum"


@pytest.mark.parametrize(
    "body",
    [
        # vLLM, when max_tokens alone is too large: compaction cannot fix it.
        b'{"error":{"message":"\'max_tokens\' (40000) cannot be greater than the'
        b' model\'s maximum context length (32768).","code":400}}',
        b'{"error":{"message":"Invalid tool schema","type":"invalid_request_error"}}',
        b'{"error":{"message":"model not found","code":404}}',
        b"Internal Server Error",
    ],
)
def test_other_errors_are_not_mistaken_for_an_overflow(body):
    assert detect_overflow(parse_upstream_error(body)) is None


def test_a_non_overflow_error_keeps_its_status_and_loses_its_secrets():
    body = (
        b'{"error":{"message":"bad key sk-live-abcdef123456 for Bearer'
        b' eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJl",'
        b'"type":"invalid_request_error"}}'
    )
    error = upstream_failure(
        parse_upstream_error(body),
        status=401,
        estimate=10,
        window=None,
        secrets=("sk-live-abcdef123456",),
    )
    assert error.status == 401
    assert error.error_type == "invalid_request_error"
    assert "sk-live" not in error.message
    assert "eyJ" not in error.message


def test_retry_hints_from_the_upstream_are_kept_on_a_rate_limit():
    error = upstream_failure(
        parse_upstream_error(b'{"error":{"message":"slow down"}}'),
        status=429,
        estimate=10,
        window=None,
        secrets=(),
        headers={"Retry-After": "7", "Set-Cookie": "nope"},
    )
    assert error.status == 429
    assert error.error_type == "rate_limit_error"
    assert error.headers == {"retry-after": "7"}
