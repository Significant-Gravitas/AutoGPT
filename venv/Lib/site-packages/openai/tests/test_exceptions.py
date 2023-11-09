import pickle

import pytest

import openai

EXCEPTION_TEST_CASES = [
    openai.InvalidRequestError(
        "message",
        "param",
        code=400,
        http_body={"test": "test1"},
        http_status="fail",
        json_body={"text": "iono some text"},
        headers={"request-id": "asasd"},
    ),
    openai.error.AuthenticationError(),
    openai.error.PermissionError(),
    openai.error.RateLimitError(),
    openai.error.ServiceUnavailableError(),
    openai.error.SignatureVerificationError("message", "sig_header?"),
    openai.error.APIConnectionError("message!", should_retry=True),
    openai.error.TryAgain(),
    openai.error.Timeout(),
    openai.error.APIError(
        message="message",
        code=400,
        http_body={"test": "test1"},
        http_status="fail",
        json_body={"text": "iono some text"},
        headers={"request-id": "asasd"},
    ),
    openai.error.OpenAIError(),
]


class TestExceptions:
    @pytest.mark.parametrize("error", EXCEPTION_TEST_CASES)
    def test_exceptions_are_pickleable(self, error) -> None:
        assert error.__repr__() == pickle.loads(pickle.dumps(error)).__repr__()
