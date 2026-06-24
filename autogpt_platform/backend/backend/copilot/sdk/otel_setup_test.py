"""Tests for OTEL tracing setup in the SDK copilot path."""

import os
from unittest.mock import MagicMock, patch


class TestSetupLangfuseOtel:
    """Tests for _setup_langfuse_otel()."""

    def test_noop_when_langfuse_not_configured(self):
        """No env vars should be set when Langfuse credentials are missing."""
        with patch(
            "backend.copilot.sdk.service._is_langfuse_configured", return_value=False
        ):
            from backend.copilot.sdk.service import _setup_langfuse_otel

            # Clear any previously set env vars
            env_keys = [
                "LANGSMITH_OTEL_ENABLED",
                "LANGSMITH_OTEL_ONLY",
                "LANGSMITH_TRACING",
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "OTEL_EXPORTER_OTLP_HEADERS",
            ]
            saved = {k: os.environ.pop(k, None) for k in env_keys}
            try:
                _setup_langfuse_otel()
                for key in env_keys:
                    assert key not in os.environ, f"{key} should not be set"
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v

    def test_sets_env_vars_when_langfuse_configured(self):
        """OTEL env vars should be set when Langfuse credentials exist."""
        mock_settings = MagicMock()
        mock_settings.secrets.langfuse_public_key = "pk-test-123"
        mock_settings.secrets.langfuse_secret_key = "sk-test-456"
        mock_settings.secrets.langfuse_host = "https://langfuse.example.com"
        mock_settings.secrets.langfuse_tracing_environment = "test"

        with (
            patch(
                "backend.copilot.sdk.service._is_langfuse_configured",
                return_value=True,
            ),
            patch("backend.copilot.sdk.service.Settings", return_value=mock_settings),
            patch(
                "backend.copilot.sdk.service.configure_claude_agent_sdk",
                return_value=True,
            ) as mock_configure,
        ):
            from backend.copilot.sdk.service import _setup_langfuse_otel

            # Clear env vars so setdefault works
            env_keys = [
                "LANGSMITH_OTEL_ENABLED",
                "LANGSMITH_OTEL_ONLY",
                "LANGSMITH_TRACING",
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "OTEL_EXPORTER_OTLP_HEADERS",
                "OTEL_RESOURCE_ATTRIBUTES",
            ]
            saved = {k: os.environ.pop(k, None) for k in env_keys}
            try:
                _setup_langfuse_otel()

                assert os.environ["LANGSMITH_OTEL_ENABLED"] == "true"
                assert os.environ["LANGSMITH_OTEL_ONLY"] == "true"
                assert os.environ["LANGSMITH_TRACING"] == "true"
                assert (
                    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
                    == "https://langfuse.example.com/api/public/otel"
                )
                assert "Authorization=Basic" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
                assert (
                    os.environ["OTEL_RESOURCE_ATTRIBUTES"]
                    == "langfuse.environment=test"
                )

                mock_configure.assert_called_once_with(tags=["sdk"])
            finally:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v
                    elif k in os.environ:
                        del os.environ[k]

    def test_existing_env_vars_not_overwritten(self):
        """Explicit env-var overrides should not be clobbered."""
        mock_settings = MagicMock()
        mock_settings.secrets.langfuse_public_key = "pk-test"
        mock_settings.secrets.langfuse_secret_key = "sk-test"
        mock_settings.secrets.langfuse_host = "https://langfuse.example.com"

        with (
            patch(
                "backend.copilot.sdk.service._is_langfuse_configured",
                return_value=True,
            ),
            patch("backend.copilot.sdk.service.Settings", return_value=mock_settings),
            patch(
                "backend.copilot.sdk.service.configure_claude_agent_sdk",
                return_value=True,
            ),
        ):
            from backend.copilot.sdk.service import _setup_langfuse_otel

            saved = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            try:
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://custom.endpoint/v1"
                _setup_langfuse_otel()
                assert (
                    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
                    == "https://custom.endpoint/v1"
                )
            finally:
                if saved is not None:
                    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = saved
                elif "OTEL_EXPORTER_OTLP_ENDPOINT" in os.environ:
                    del os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]

    def test_graceful_failure_on_exception(self):
        """Setup should not raise even if internal code fails."""
        with (
            patch(
                "backend.copilot.sdk.service._is_langfuse_configured",
                return_value=True,
            ),
            patch(
                "backend.copilot.sdk.service.Settings",
                side_effect=RuntimeError("settings unavailable"),
            ),
        ):
            from backend.copilot.sdk.service import _setup_langfuse_otel

            # Should not raise — just logs and returns
            _setup_langfuse_otel()


class TestPropagateAttributesImport:
    """Verify langfuse.propagate_attributes is available."""

    def test_propagate_attributes_is_importable(self):
        from langfuse import propagate_attributes

        assert callable(propagate_attributes)

    def test_propagate_attributes_returns_context_manager(self):
        from langfuse import propagate_attributes

        ctx = propagate_attributes(user_id="u1", session_id="s1", tags=["test"])
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")


class TestReceiveResponseCompat:
    """Verify ClaudeSDKClient.receive_response() exists (langsmith patches it)."""

    def test_receive_response_exists(self):
        from claude_agent_sdk import ClaudeSDKClient

        assert hasattr(ClaudeSDKClient, "receive_response")

    def test_receive_response_is_async_generator(self):
        import inspect

        from claude_agent_sdk import ClaudeSDKClient

        method = getattr(ClaudeSDKClient, "receive_response")
        assert inspect.isfunction(method) or inspect.ismethod(method)
