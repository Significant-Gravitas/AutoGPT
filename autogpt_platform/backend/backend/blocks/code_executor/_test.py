from typing import Any

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="e2b",
    api_key=SecretStr("mock-e2b-api-key"),
    title="Mock E2B API key",
    expires_at=None,
)


def mock_execute_code(
    *args: Any, **kwargs: Any
) -> tuple[list, str, str, str, str, list]:
    """Stub for `execute_code` used by the blocks' test_mock.

    Returns the (results, text, stdout, stderr, sandbox_id, files) tuple,
    echoing back any provided `sandbox_id` so the step block's test sees it.
    """
    sandbox_id = kwargs.get("sandbox_id") or "sandbox_id"
    return [], "Hello World", "Hello World\n", "", sandbox_id, []


TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}
