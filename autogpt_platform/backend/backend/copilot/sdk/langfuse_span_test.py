"""Unit tests for _LangfuseSDKSpan and sdk/service.py helper functions.

All tests run without network access, real Langfuse credentials, or a running
database — every external dependency is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.copilot.sdk.service import (
    _build_sdk_env,
    _LangfuseSDKSpan,
    _make_sdk_cwd,
    _resolve_sdk_model,
)

# ---------------------------------------------------------------------------
# _resolve_sdk_model
# ---------------------------------------------------------------------------


def test_resolve_sdk_model_uses_claude_agent_model():
    with patch("backend.copilot.sdk.service.config") as mock_cfg:
        mock_cfg.claude_agent_model = "claude-opus-4-6"
        mock_cfg.model = "anthropic/claude-sonnet-4-6"
        assert _resolve_sdk_model() == "claude-opus-4-6"


def test_resolve_sdk_model_strips_provider_prefix():
    with patch("backend.copilot.sdk.service.config") as mock_cfg:
        mock_cfg.claude_agent_model = ""
        mock_cfg.model = "anthropic/claude-opus-4.6"
        assert _resolve_sdk_model() == "claude-opus-4.6"


def test_resolve_sdk_model_no_prefix():
    with patch("backend.copilot.sdk.service.config") as mock_cfg:
        mock_cfg.claude_agent_model = ""
        mock_cfg.model = "claude-haiku-3"
        assert _resolve_sdk_model() == "claude-haiku-3"


# ---------------------------------------------------------------------------
# _build_sdk_env
# ---------------------------------------------------------------------------


def test_build_sdk_env_with_credentials():
    with patch("backend.copilot.sdk.service.config") as mock_cfg:
        mock_cfg.api_key = "sk-test"
        mock_cfg.base_url = "https://openrouter.ai/api/v1"
        env = _build_sdk_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
    assert env["ANTHROPIC_API_KEY"] == ""


def test_build_sdk_env_strips_v1_suffix():
    with patch("backend.copilot.sdk.service.config") as mock_cfg:
        mock_cfg.api_key = "key"
        mock_cfg.base_url = "https://custom.host/v1"
        env = _build_sdk_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://custom.host"


def test_build_sdk_env_no_credentials():
    with patch("backend.copilot.sdk.service.config") as mock_cfg:
        mock_cfg.api_key = ""
        mock_cfg.base_url = ""
        env = _build_sdk_env()
    assert env == {}


def test_build_sdk_env_invalid_base_url():
    with patch("backend.copilot.sdk.service.config") as mock_cfg:
        mock_cfg.api_key = "key"
        mock_cfg.base_url = "not-a-url"
        env = _build_sdk_env()
    assert env == {}


# ---------------------------------------------------------------------------
# _make_sdk_cwd
# ---------------------------------------------------------------------------


def test_make_sdk_cwd_rejects_traversal():
    with patch(
        "backend.copilot.sdk.service.make_session_path",
        return_value="/etc/passwd",
    ):
        with pytest.raises(ValueError, match="escaped prefix"):
            _make_sdk_cwd("anything")


def test_make_sdk_cwd_valid_session():
    import os

    from backend.copilot.sdk.service import _SDK_CWD_PREFIX

    with patch(
        "backend.copilot.sdk.service.make_session_path",
        return_value=os.path.join(_SDK_CWD_PREFIX, "test-session"),
    ):
        cwd = _make_sdk_cwd("test-session")
    assert cwd.startswith(_SDK_CWD_PREFIX)


# ---------------------------------------------------------------------------
# _LangfuseSDKSpan — no-op when Langfuse is not configured
# ---------------------------------------------------------------------------


@pytest.fixture()
def langfuse_disabled():
    with patch(
        "backend.copilot.sdk.service._is_langfuse_configured", return_value=False
    ):
        yield


def test_span_noop_when_langfuse_disabled(langfuse_disabled):
    span = _LangfuseSDKSpan(
        session_id="s1", user_id="u1", model="claude", input="hello"
    )
    assert span._lf is None
    assert span._ctx is None
    # All methods are safe no-ops
    span.update_usage({"input_tokens": 10, "output_tokens": 5}, 0.001)
    span.finish("some output")
    span.close()


# ---------------------------------------------------------------------------
# _LangfuseSDKSpan — active when Langfuse is configured
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_langfuse():
    """Returns (mock_client, mock_ctx) and patches the module-level getter."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    mock_lf = MagicMock()
    mock_lf.start_as_current_observation.return_value = mock_ctx

    with patch(
        "backend.copilot.sdk.service._is_langfuse_configured", return_value=True
    ), patch("backend.copilot.sdk.service._get_langfuse_client", return_value=mock_lf):
        yield mock_lf, mock_ctx


def test_span_init_calls_langfuse(mock_langfuse):
    mock_lf, mock_ctx = mock_langfuse
    _LangfuseSDKSpan(session_id="s1", user_id="u1", model="claude-haiku", input="hi")
    mock_lf.start_as_current_observation.assert_called_once_with(
        name="copilot-sdk-session",
        as_type="generation",
        model="claude-haiku",
        input="hi",
    )
    mock_ctx.__enter__.assert_called_once()
    mock_lf.update_current_trace.assert_called_once_with(
        session_id="s1", user_id="u1", tags=["sdk"]
    )


def test_span_update_usage(mock_langfuse):
    mock_lf, _ = mock_langfuse
    span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    span.update_usage({"input_tokens": 100, "output_tokens": 50}, 0.002)
    mock_lf.update_current_generation.assert_called_once_with(
        usage_details={"input": 100, "output": 50},
        cost_details={"total": 0.002},
    )


def test_span_update_usage_no_cost(mock_langfuse):
    mock_lf, _ = mock_langfuse
    span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    span.update_usage({"input_tokens": 10, "output_tokens": 5}, None)
    mock_lf.update_current_generation.assert_called_once_with(
        usage_details={"input": 10, "output": 5},
        cost_details=None,
    )


def test_span_update_usage_none_usage(mock_langfuse):
    mock_lf, _ = mock_langfuse
    span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    span.update_usage(None, None)
    mock_lf.update_current_generation.assert_called_once_with(
        usage_details={"input": 0, "output": 0},
        cost_details=None,
    )


def test_span_finish(mock_langfuse):
    mock_lf, _ = mock_langfuse
    span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    span.finish("final answer")
    mock_lf.update_current_trace.assert_called_with(output="final answer")


def test_span_close_no_exc_info(mock_langfuse):
    _, mock_ctx = mock_langfuse
    span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    span.close()
    mock_ctx.__exit__.assert_called_once_with(None, None, None)


def test_span_close_passes_exc_info(mock_langfuse):
    """close(sys.exc_info()) forwards exception details to Langfuse."""
    _, mock_ctx = mock_langfuse
    span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    exc = ValueError("boom")
    exc_info = (type(exc), exc, None)
    span.close(exc_info)
    mock_ctx.__exit__.assert_called_once_with(*exc_info)


def test_span_init_failure_is_silent():
    """If Langfuse init raises, span degrades to no-op without propagating."""
    with patch(
        "backend.copilot.sdk.service._is_langfuse_configured", return_value=True
    ), patch(
        "backend.copilot.sdk.service._get_langfuse_client",
        side_effect=RuntimeError("connection refused"),
    ):
        span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    assert span._lf is None
    # Safe to call after failed init
    span.update_usage({}, None)
    span.finish(None)
    span.close()


def test_span_init_ctx_leaked_when_update_trace_fails():
    """If __enter__ succeeds but update_current_trace raises, __exit__ is called."""
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    mock_lf = MagicMock()
    mock_lf.start_as_current_observation.return_value = mock_ctx
    mock_lf.update_current_trace.side_effect = RuntimeError("trace update failed")

    with patch(
        "backend.copilot.sdk.service._is_langfuse_configured", return_value=True
    ), patch("backend.copilot.sdk.service._get_langfuse_client", return_value=mock_lf):
        span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)

    # Span must degrade to no-op
    assert span._lf is None
    assert span._ctx is None
    # The already-entered context must have been exited to avoid a leak
    mock_ctx.__exit__.assert_called_once()


def test_span_close_noop_when_no_ctx(langfuse_disabled):
    span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    span.close()  # must not raise


def test_span_ctx_none_from_langfuse(mock_langfuse):
    """start_as_current_observation returning None shouldn't crash __enter__."""
    mock_lf, _ = mock_langfuse
    mock_lf.start_as_current_observation.return_value = None
    span = _LangfuseSDKSpan(session_id="s1", user_id=None, model=None, input=None)
    # ctx is None so close must be safe
    span.close()
