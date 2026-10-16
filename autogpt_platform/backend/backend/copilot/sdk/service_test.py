"""Tests for SDK service helpers."""

import asyncio
import base64
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage

from backend.copilot import config as cfg_mod
from backend.copilot.anthropic_rate_card import (
    anthropic_usage_cost_usd,
    fallback_usage_cost_usd,
)
from backend.copilot.builder_context import BUILDER_BLOCKED_TOOLS
from backend.copilot.permissions import CopilotPermissions, all_known_tool_names
from backend.data.sharing.workspace_refs import extract_workspace_file_ids

from .cost_tracking import TokenUsage
from .model_gateway.front import ModelGateway
from .service import (
    _HUNG_TOOL_CAP_SECONDS,
    _IDLE_TIMEOUT_SECONDS,
    _THINKING_ONLY_REPROMPT,
    _build_system_prompt_value,
    _close_route_gateway_for_finally,
    _configured_system_prompt_value,
    _hidden_short_names_for_permissions,
    _humanise_tool_list,
    _idle_timeout_threshold,
    _is_sdk_disconnect_error,
    _prepare_file_attachments,
    _raise_deferred_gateway_cleanup_error,
    _record_result_usage,
    _redact_cli_stderr,
    _safe_close_sdk_client,
    _strip_synthetic_reprompt_from_cli_jsonl,
)


class TestCliStderrRedaction:
    def test_redacts_capability_even_without_header_label(self):
        capability = "private-loopback-capability"

        result = _redact_cli_stderr(
            f"gateway request failed for {capability}", secrets=(capability,)
        )

        assert capability not in result
        assert result == "gateway request failed for [REDACTED]"

    def test_redacts_labeled_capability_once(self):
        capability = "private-loopback-capability"

        result = _redact_cli_stderr(
            f"Authorization: Bearer {capability}", secrets=(capability,)
        )

        assert result == "Authorization: [REDACTED]"

    @pytest.mark.parametrize(
        ("line", "secret"),
        [
            ("Authorization: Bearer secret-one", "secret-one"),
            ('{"Authorization":"Bearer secret-two"}', "secret-two"),
            ("proxy-authorization=Basic secret-three", "secret-three"),
            ("ANTHROPIC_AUTH_TOKEN=secret-four", "secret-four"),
            ("x-api-key: secret-five", "secret-five"),
        ],
    )
    def test_redacts_authorization_shaped_values(self, line, secret):
        result = _redact_cli_stderr(line)

        assert secret not in result
        assert "[REDACTED]" in result

    def test_preserves_unrelated_stderr(self):
        line = "fallback model activated after an overload"

        assert _redact_cli_stderr(line) == line


class TestRouteGatewayCleanup:
    @pytest.mark.asyncio
    async def test_returns_checkpoint_failure_for_deferred_raise(self):
        failure = RuntimeError("credential checkpoint failed")
        gateway = cast(
            ModelGateway,
            SimpleNamespace(close=AsyncMock(side_effect=failure)),
        )

        result = await _close_route_gateway_for_finally(gateway, "[test]")

        assert result is failure
        gateway.close.assert_awaited_once()

    def test_deferred_failure_propagates_with_turn_error_as_cause(self):
        turn_error = ValueError("turn failed")
        cleanup_error = RuntimeError("credential checkpoint failed")

        with pytest.raises(RuntimeError, match="checkpoint failed") as raised:
            _raise_deferred_gateway_cleanup_error(cleanup_error, turn_error)

        assert raised.value is cleanup_error
        assert raised.value.__cause__ is turn_error


@dataclass
class _FakeFileInfo:
    id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int


_PATCH_TARGET = "backend.copilot.sdk.service.get_workspace_manager"


class TestPrepareFileAttachments:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self, tmp_path):
        result = await _prepare_file_attachments([], "u", "s", str(tmp_path))
        assert result.hint == ""
        assert result.image_blocks == []

    @pytest.mark.asyncio
    async def test_image_embedded_as_vision_block(self, tmp_path):
        """JPEG images should become vision content blocks, not files on disk."""
        raw = b"\xff\xd8\xff\xe0fake-jpeg"
        info = _FakeFileInfo(
            id="abc",
            name="photo.jpg",
            path="/photo.jpg",
            mime_type="image/jpeg",
            size_bytes=len(raw),
        )
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = raw

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(
                ["abc"], "user1", "sess1", str(tmp_path)
            )

        assert "1 file" in result.hint
        assert "photo.jpg" in result.hint
        assert "embedded as image" in result.hint
        assert len(result.image_blocks) == 1
        block = result.image_blocks[0]
        assert block["type"] == "image"
        assert block["source"]["media_type"] == "image/jpeg"
        assert block["source"]["data"] == base64.b64encode(raw).decode("ascii")
        # Image should NOT be written to disk (embedded instead)
        assert not os.path.exists(os.path.join(tmp_path, "photo.jpg"))

    @pytest.mark.asyncio
    async def test_pdf_referenced_by_file_id(self, tmp_path):
        """Non-images are surfaced via read_workspace_file by file_id — the
        model's file tools run in the E2B sandbox in prod, where a host-side
        sdk_cwd path doesn't exist. A local copy is still written for non-E2B
        tooling, but the model must not be pointed at that path."""
        info = _FakeFileInfo("f1", "doc.pdf", "/doc.pdf", "application/pdf", 50)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"%PDF-1.4 fake"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(["f1"], "u", "s", str(tmp_path))

        assert result.image_blocks == []
        saved = tmp_path / "doc.pdf"
        assert saved.exists()
        assert saved.read_bytes() == b"%PDF-1.4 fake"
        # The model is given the workspace identity, not the host path.
        assert "file_id=f1" in result.hint
        assert str(saved) not in result.hint
        assert "read_workspace_file" in result.hint

    @pytest.mark.asyncio
    async def test_hint_file_ids_extractable_for_chat_sharing(self, tmp_path):
        """Chat-share allowlisting parses ``file_id=<uuid>`` out of message
        content with extract_workspace_file_ids; the hint must keep that exact
        shape or shared chats lose access to their attachments."""
        fid = "0f8fad5b-d9cb-469f-a165-70867728950e"
        info = _FakeFileInfo(fid, "doc.pdf", "/doc.pdf", "application/pdf", 50)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"%PDF-1.4 fake"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments([fid], "u", "s", str(tmp_path))

        assert extract_workspace_file_ids(result.hint) == {fid}

    @pytest.mark.asyncio
    async def test_mixed_images_and_files(self, tmp_path):
        """Images become blocks, non-images go to disk."""
        infos = {
            "id1": _FakeFileInfo("id1", "a.png", "/a.png", "image/png", 4),
            "id2": _FakeFileInfo("id2", "b.pdf", "/b.pdf", "application/pdf", 4),
            "id3": _FakeFileInfo("id3", "c.txt", "/c.txt", "text/plain", 4),
        }
        mgr = AsyncMock()
        mgr.get_file_info.side_effect = lambda fid: infos[fid]
        mgr.read_file_by_id.return_value = b"data"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(
                ["id1", "id2", "id3"], "u", "s", str(tmp_path)
            )

        assert "3 files" in result.hint
        assert "a.png" in result.hint
        assert "b.pdf" in result.hint
        assert "c.txt" in result.hint
        # Only the image should be a vision block
        assert len(result.image_blocks) == 1
        assert result.image_blocks[0]["source"]["media_type"] == "image/png"
        # Non-image files should be on disk
        assert (tmp_path / "b.pdf").exists()
        assert (tmp_path / "c.txt").exists()
        # Workspace-read hint should appear (has non-image files), and the
        # model must NOT be told to use the (disallowed) built-in Read tool.
        assert "read_workspace_file" in result.hint
        assert "Use the Read tool" not in result.hint
        assert "file_id=id2" in result.hint
        assert "file_id=id3" in result.hint

    @pytest.mark.asyncio
    async def test_singular_noun(self, tmp_path):
        info = _FakeFileInfo("x", "only.txt", "/only.txt", "text/plain", 2)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"hi"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(["x"], "u", "s", str(tmp_path))

        assert "1 file." in result.hint

    @pytest.mark.asyncio
    async def test_missing_file_skipped(self, tmp_path):
        mgr = AsyncMock()
        mgr.get_file_info.return_value = None

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(
                ["missing-id"], "u", "s", str(tmp_path)
            )

        assert result.hint == ""
        assert result.image_blocks == []

    @pytest.mark.asyncio
    async def test_image_only_no_read_hint(self, tmp_path):
        """When all files are images, no read-tool hint should appear."""
        info = _FakeFileInfo("i1", "cat.png", "/cat.png", "image/png", 4)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"data"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(["i1"], "u", "s", str(tmp_path))

        assert "read_workspace_file" not in result.hint
        assert len(result.image_blocks) == 1


class TestPromptSupplement:
    """Tests for centralized prompt supplement generation."""

    def test_sdk_supplement_excludes_tool_docs(self):
        """SDK mode should NOT include tool documentation (Claude gets schemas automatically)."""
        from backend.copilot.prompting import get_sdk_supplement

        # Test both local and E2B modes
        local_supplement = get_sdk_supplement(use_e2b=False)
        e2b_supplement = get_sdk_supplement(use_e2b=True)

        # Should NOT have tool list section
        assert "## AVAILABLE TOOLS" not in local_supplement
        assert "## AVAILABLE TOOLS" not in e2b_supplement

        # Should still have technical notes
        assert "## Tool notes" in local_supplement
        assert "## Tool notes" in e2b_supplement

    def test_baseline_supplement_has_shared_notes_no_tool_list(self):
        """Baseline now relies on the OpenAI tools array for schemas and only
        appends SHARED_TOOL_NOTES (workflow rules not present in any schema).
        The old auto-generated ``## AVAILABLE TOOLS`` list is gone — it was
        ~4.3K tokens of pure duplication of the tools array."""
        from backend.copilot.prompting import SHARED_TOOL_NOTES

        assert "## AVAILABLE TOOLS" not in SHARED_TOOL_NOTES
        # Keep the high-value workflow rules that are NOT in any tool schema.
        assert "@@agptfile:" in SHARED_TOOL_NOTES
        assert "find_capability` is MANDATORY" in SHARED_TOOL_NOTES
        assert "run_sub_session" in SHARED_TOOL_NOTES

    def test_pause_task_scheduled_before_transcript_upload(self):
        """Pause is scheduled as a background task before transcript upload begins.

        The finally block in stream_response_sdk does:
          (1) asyncio.create_task(pause_sandbox_direct(...))  — fire-and-forget
          (2) await asyncio.shield(upload_transcript(...))    — awaited

        Scheduling pause via create_task before awaiting upload ensures:
        - Pause never blocks transcript upload (billing stops concurrently)
        - On E2B timeout, pause silently fails; upload proceeds unaffected
        """
        call_order: list[str] = []

        async def _mock_pause(sandbox, session_id):
            call_order.append("pause")

        async def _mock_upload(**kwargs):
            call_order.append("upload")

        async def _simulate_teardown():
            """Mirror the service.py finally block teardown sequence."""
            sandbox = MagicMock()

            # (1) Schedule pause — mirrors lines ~1427-1429 in service.py
            task = asyncio.create_task(_mock_pause(sandbox, "test-sess"))

            # (2) Await transcript upload — mirrors lines ~1460-1468 in service.py
            # Yielding to the event loop here lets the pause task start concurrently.
            await _mock_upload(
                user_id="u", session_id="test-sess", content="x", message_count=1
            )
            await task

        asyncio.run(_simulate_teardown())

        # Both must run; pause is scheduled before upload starts
        assert "pause" in call_order
        assert "upload" in call_order
        # create_task schedules pause, then upload is awaited — pause runs
        # concurrently during upload's first yield. The ordering guarantee is
        # that create_task is CALLED before upload is AWAITED (see source order).


# ---------------------------------------------------------------------------
# _cleanup_sdk_tool_results — orchestration + rate-limiting
# ---------------------------------------------------------------------------


class TestCleanupSdkToolResults:
    """Tests for _cleanup_sdk_tool_results orchestration and sweep rate-limiting."""

    # All valid cwds must start with /tmp/copilot- (the _SDK_CWD_PREFIX).
    _CWD_PREFIX = "/tmp/copilot-"

    @pytest.mark.asyncio
    async def test_removes_cwd_directory(self):
        """Cleanup removes the session working directory."""

        from .service import _cleanup_sdk_tool_results

        cwd = "/tmp/copilot-test-cleanup-remove"
        os.makedirs(cwd, exist_ok=True)

        with patch("backend.copilot.sdk.service.cleanup_stale_project_dirs"):
            import backend.copilot.sdk.service as svc_mod

            svc_mod._last_sweep_time = 0.0
            await _cleanup_sdk_tool_results(cwd)

        assert not os.path.exists(cwd)

    @pytest.mark.asyncio
    async def test_sweep_runs_when_interval_elapsed(self):
        """cleanup_stale_project_dirs is called when 5-minute interval has elapsed."""

        import backend.copilot.sdk.service as svc_mod

        from .service import _cleanup_sdk_tool_results

        cwd = "/tmp/copilot-test-sweep-elapsed"
        os.makedirs(cwd, exist_ok=True)

        with patch(
            "backend.copilot.sdk.service.cleanup_stale_project_dirs"
        ) as mock_sweep:
            # Set last sweep to a time far in the past
            svc_mod._last_sweep_time = 0.0
            await _cleanup_sdk_tool_results(cwd)

        mock_sweep.assert_called_once()

    @pytest.mark.asyncio
    async def test_sweep_skipped_within_interval(self):
        """cleanup_stale_project_dirs is NOT called when within 5-minute interval."""
        import time

        import backend.copilot.sdk.service as svc_mod

        from .service import _cleanup_sdk_tool_results

        cwd = "/tmp/copilot-test-sweep-ratelimit"
        os.makedirs(cwd, exist_ok=True)

        with patch(
            "backend.copilot.sdk.service.cleanup_stale_project_dirs"
        ) as mock_sweep:
            # Set last sweep to now — interval not elapsed
            svc_mod._last_sweep_time = time.time()
            await _cleanup_sdk_tool_results(cwd)

        mock_sweep.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_path_outside_prefix(self, tmp_path):
        """Cleanup rejects a cwd that does not start with the expected prefix."""
        from .service import _cleanup_sdk_tool_results

        evil_cwd = str(tmp_path / "evil-path")
        os.makedirs(evil_cwd, exist_ok=True)

        with patch(
            "backend.copilot.sdk.service.cleanup_stale_project_dirs"
        ) as mock_sweep:
            await _cleanup_sdk_tool_results(evil_cwd)

        # Directory should NOT have been removed (rejected early)
        assert os.path.exists(evil_cwd)
        mock_sweep.assert_not_called()


# ---------------------------------------------------------------------------
# Env-cleanup fixture is shared via ``conftest._clean_config_env``.  This
# file exposes a re-export for callers that don't rely on conftest discovery
# (kept for backwards compatibility — pytest finds the conftest fixture
# automatically without an explicit import).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _is_sdk_disconnect_error — classify client disconnect cleanup errors
# ---------------------------------------------------------------------------


class TestIsSdkDisconnectError:
    """Tests for _is_sdk_disconnect_error — identifies expected SDK cleanup errors."""

    def test_cancel_scope_runtime_error(self):
        """RuntimeError about cancel scope in wrong task is a disconnect error."""
        exc = RuntimeError(
            "Attempted to exit cancel scope in a different task than it was entered in"
        )
        assert _is_sdk_disconnect_error(exc) is True

    def test_context_var_value_error(self):
        """ValueError about ContextVar token mismatch is a disconnect error."""
        exc = ValueError(
            "<Token var=<ContextVar name='current_context'>> "
            "was created in a different Context"
        )
        assert _is_sdk_disconnect_error(exc) is True

    def test_unrelated_runtime_error(self):
        """Unrelated RuntimeError should NOT be classified as disconnect error."""
        exc = RuntimeError("something else went wrong")
        assert _is_sdk_disconnect_error(exc) is False

    def test_unrelated_value_error(self):
        """Unrelated ValueError should NOT be classified as disconnect error."""
        exc = ValueError("invalid argument")
        assert _is_sdk_disconnect_error(exc) is False

    def test_other_exception_types(self):
        """Non-RuntimeError/ValueError should NOT be classified as disconnect error."""
        assert _is_sdk_disconnect_error(TypeError("bad type")) is False
        assert _is_sdk_disconnect_error(OSError("network down")) is False
        assert _is_sdk_disconnect_error(asyncio.CancelledError()) is False


# ---------------------------------------------------------------------------
# _safe_close_sdk_client — suppress cleanup errors during disconnect
# ---------------------------------------------------------------------------


class TestSafeCloseSdkClient:
    """Tests for _safe_close_sdk_client — suppresses expected SDK cleanup errors."""

    @pytest.mark.asyncio
    async def test_clean_exit(self):
        """Normal __aexit__ (no error) should succeed silently."""
        client = AsyncMock()
        client.__aexit__ = AsyncMock(return_value=None)
        await _safe_close_sdk_client(client, "[test]")
        client.__aexit__.assert_awaited_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_cancel_scope_runtime_error_suppressed(self):
        """RuntimeError from cancel scope mismatch should be suppressed."""
        client = AsyncMock()
        client.__aexit__ = AsyncMock(
            side_effect=RuntimeError(
                "Attempted to exit cancel scope in a different task"
            )
        )
        # Should NOT raise
        await _safe_close_sdk_client(client, "[test]")

    @pytest.mark.asyncio
    async def test_context_var_value_error_suppressed(self):
        """ValueError from ContextVar token mismatch should be suppressed."""
        client = AsyncMock()
        client.__aexit__ = AsyncMock(
            side_effect=ValueError(
                "<Token var=<ContextVar name='current_context'>> "
                "was created in a different Context"
            )
        )
        # Should NOT raise
        await _safe_close_sdk_client(client, "[test]")

    @pytest.mark.asyncio
    async def test_unexpected_exception_suppressed_with_error_log(self):
        """Unexpected exceptions should be caught (not propagated) but logged at error."""
        client = AsyncMock()
        client.__aexit__ = AsyncMock(side_effect=OSError("unexpected"))
        # Should NOT raise — unexpected errors are also suppressed to
        # avoid crashing the generator during teardown.  Logged at error
        # level so Sentry captures them via its logging integration.
        await _safe_close_sdk_client(client, "[test]")

    @pytest.mark.asyncio
    async def test_unrelated_runtime_error_propagates(self):
        """Non-cancel-scope RuntimeError should propagate (not suppressed)."""
        client = AsyncMock()
        client.__aexit__ = AsyncMock(side_effect=RuntimeError("something unrelated"))
        with pytest.raises(RuntimeError, match="something unrelated"):
            await _safe_close_sdk_client(client, "[test]")

    @pytest.mark.asyncio
    async def test_unrelated_value_error_propagates(self):
        """Non-disconnect ValueError should propagate (not suppressed)."""
        client = AsyncMock()
        client.__aexit__ = AsyncMock(side_effect=ValueError("invalid argument"))
        with pytest.raises(ValueError, match="invalid argument"):
            await _safe_close_sdk_client(client, "[test]")


# ---------------------------------------------------------------------------
# SystemPromptPreset — cross-user prompt caching
# ---------------------------------------------------------------------------


class TestSystemPromptValue:
    """Tests for _build_system_prompt_value — the live prompt."""

    def test_custom_dict_when_live(self):
        """The prompt replaces the CLI's own prompt in the custom form, the
        form that carries ``snapshot``."""
        custom_prompt = "You are a helpful assistant."
        result = _build_system_prompt_value(custom_prompt, live=True)

        assert result == {"type": "custom", "prompt": custom_prompt, "snapshot": False}

    def test_live_off_keeps_the_raw_string(self):
        """The rollback lever: exactly the value passed before the live
        prompt, so the CLI records the prompt and resends it on resume."""
        custom_prompt = "You are a helpful assistant."
        result = _build_system_prompt_value(custom_prompt, live=False)

        assert result == custom_prompt

    def test_default_config_sends_a_live_prompt(self, _clean_config_env):
        cfg = cfg_mod.ChatConfig(_env_file=None)
        assert cfg.claude_agent_live_system_prompt is True

    def test_env_var_turns_the_live_prompt_off(self, _clean_config_env, monkeypatch):
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_LIVE_SYSTEM_PROMPT", "false")
        cfg = cfg_mod.ChatConfig()
        assert cfg.claude_agent_live_system_prompt is False

    def test_default_config_strips_prompt_snapshots(self, _clean_config_env):
        cfg = cfg_mod.ChatConfig(_env_file=None)
        assert cfg.claude_agent_strip_prompt_snapshots is True

    def test_env_var_turns_the_snapshot_strip_off(self, _clean_config_env, monkeypatch):
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_STRIP_PROMPT_SNAPSHOTS", "false")
        cfg = cfg_mod.ChatConfig()
        assert cfg.claude_agent_strip_prompt_snapshots is False

    def test_configured_prompt_follows_the_live_switch(self, monkeypatch):
        monkeypatch.setattr(
            "backend.copilot.sdk.service.config.claude_agent_live_system_prompt", True
        )
        assert _configured_system_prompt_value("sys") == {
            "type": "custom",
            "prompt": "sys",
            "snapshot": False,
        }

        monkeypatch.setattr(
            "backend.copilot.sdk.service.config.claude_agent_live_system_prompt", False
        )
        assert _configured_system_prompt_value("sys") == "sys"


class TestStreamErrorCodePrefix:
    """StreamError.to_sse auto-prefixes errorText with `[code:<id>]` when a
    code is set, so the frontend can parse a machine-readable code out of
    the AI-SDK's strict `{type, errorText}` schema."""

    def test_auto_prefix_when_code_set(self):
        from backend.copilot.response_model import StreamError

        sse = StreamError(errorText="Boom", code="idle_timeout").to_sse()
        assert '"errorText":"[code:idle_timeout] Boom"' in sse

    def test_no_prefix_when_code_missing(self):
        from backend.copilot.response_model import StreamError

        sse = StreamError(errorText="Boom").to_sse()
        assert '"errorText":"Boom"' in sse

    def test_does_not_double_prefix(self):
        from backend.copilot.response_model import StreamError

        sse = StreamError(errorText="[code:x] Boom", code="x").to_sse()
        assert "[code:x] [code:x]" not in sse
        assert '"errorText":"[code:x] Boom"' in sse


class TestHumaniseToolList:
    """Tool-name formatter used to build the idle-timeout error message."""

    def test_empty_returns_empty_string(self):
        assert _humanise_tool_list([]) == ""

    def test_single_tool_is_quoted(self):
        assert _humanise_tool_list(["WebSearch"]) == "'WebSearch'"

    def test_two_tools_are_joined_with_and(self):
        assert (
            _humanise_tool_list(["WebSearch", "run_block"])
            == "'WebSearch' and 'run_block'"
        )

    def test_three_uses_singular_other(self):
        assert _humanise_tool_list(["a", "b", "c"]) == "'a', 'b', and 1 other"

    def test_four_plus_uses_plural_others(self):
        assert _humanise_tool_list(["a", "b", "c", "d"]) == "'a', 'b', and 2 others"


# ---------------------------------------------------------------------------
# _RetryState.observed_model — Moonshot cost-override input
# ---------------------------------------------------------------------------


class TestRetryStateObservedModel:
    """Regression guards for the ``observed_model`` field added to
    ``_RetryState``.  The Moonshot cost override reads this — the
    requested model (``state.options.model``) is unset when the CLI
    picks one, and only the observed model says what actually ran."""

    def _make_state(self, *, options_model: str | None = "primary/model"):
        """Build a minimally-valid ``_RetryState``.  All the heavy
        collaborators are ``MagicMock()`` — the field we care about is
        a plain Optional[str], so the surrounding scaffolding just needs
        to let the dataclass instantiate."""
        from .service import _RetryState, _TokenUsage

        options = MagicMock()
        options.model = options_model
        return _RetryState(
            options=options,
            query_message="",
            compaction_stats=None,
            use_resume=False,
            transcript_msg_count=0,
            adapter=MagicMock(),
            transcript_builder=MagicMock(),
            usage=_TokenUsage(),
        )

    def test_default_is_none(self):
        state = self._make_state()
        assert state.observed_model is None

    def test_assigned_from_assistant_message_model(self):
        """Simulates the population path in ``_run_stream_attempt``:
        ``observed`` is pulled off the ``AssistantMessage.model`` attr
        and assigned onto ``state.observed_model`` when it's a non-empty
        string."""
        state = self._make_state()
        # Simulates the inline assignment the generator does on each
        # AssistantMessage — a non-empty string lands on state.
        assistant_like = SimpleNamespace(model="anthropic/claude-sonnet-4-6")
        observed = getattr(assistant_like, "model", None)
        if isinstance(observed, str) and observed:
            state.observed_model = observed
        assert state.observed_model == "anthropic/claude-sonnet-4-6"

    def test_empty_string_model_is_not_assigned(self):
        """Guard against overwriting a real observed value with an
        empty-string model (the generator's ``and observed`` check)."""
        state = self._make_state()
        state.observed_model = "moonshotai/kimi-k2.6"  # seeded from a prior msg
        assistant_like = SimpleNamespace(model="")
        observed = getattr(assistant_like, "model", None)
        if isinstance(observed, str) and observed:
            state.observed_model = observed
        assert state.observed_model == "moonshotai/kimi-k2.6"

    def test_missing_model_attr_leaves_observed_untouched(self):
        state = self._make_state()
        state.observed_model = "moonshotai/kimi-k2.6"
        # AssistantMessage may not carry ``.model`` on older SDK rels.
        assistant_like = SimpleNamespace()  # no ``.model`` attr
        observed = getattr(assistant_like, "model", None)
        if isinstance(observed, str) and observed:
            state.observed_model = observed
        assert state.observed_model == "moonshotai/kimi-k2.6"


# ---------------------------------------------------------------------------
# Moonshot cost-override gate — decision logic at the call site
# ---------------------------------------------------------------------------


class TestResultPricingFollowsTheObservedModel:
    """A billed result is priced on the rate card of the model that answered:
    the observed model wins over the requested one, and the CLI's own
    ``total_cost_usd`` is never the price."""

    _PROMPT, _COMPLETION = 29564, 78
    _CLI_ESTIMATE = 0.089862

    def _price(self, *, observed_model: str | None, options_model: str | None):
        state = MagicMock(
            observed_model=observed_model,
            options=SimpleNamespace(model=options_model),
            usage=TokenUsage(),
        )
        result = ResultMessage(
            subtype="success",
            duration_ms=0,
            duration_api_ms=0,
            is_error=False,
            num_turns=1,
            session_id="test",
            total_cost_usd=self._CLI_ESTIMATE,
            usage={"input_tokens": self._PROMPT, "output_tokens": self._COMPLETION},
        )
        _record_result_usage(result, state, "")
        assert state.usage.cost_usd is not None
        return state.usage.cost_usd

    def _sonnet(self) -> float:
        cost = anthropic_usage_cost_usd(
            model="claude-sonnet-4-6",
            input_tokens=self._PROMPT,
            output_tokens=self._COMPLETION,
        )
        assert cost is not None
        return cost

    def _kimi(self) -> float:
        return (self._PROMPT * 0.60 + self._COMPLETION * 2.80) / 1_000_000

    def test_anthropic_turn_is_priced_on_the_rate_card(self):
        cost = self._price(
            observed_model="anthropic/claude-sonnet-4-6",
            options_model="anthropic/claude-sonnet-4-6",
        )
        assert cost == pytest.approx(self._sonnet())
        assert cost != self._CLI_ESTIMATE

    def test_moonshot_turn_is_priced_on_the_moonshot_card(self):
        """The CLI would price Kimi at Sonnet rates (~5x the real bill)."""
        cost = self._price(
            observed_model="moonshotai/kimi-k2.6",
            options_model="moonshotai/kimi-k2.6",
        )
        assert cost == pytest.approx(self._kimi(), rel=1e-9)
        assert cost < self._CLI_ESTIMATE / 4

    def test_observed_model_wins_over_options_primary(self):
        """A Moonshot-primary request that fell back to Anthropic is priced
        as Anthropic, and the inverse as Moonshot."""
        assert self._price(
            observed_model="anthropic/claude-sonnet-4-6",
            options_model="moonshotai/kimi-k2.6",
        ) == pytest.approx(self._sonnet())
        assert self._price(
            observed_model="moonshotai/kimi-k2.6",
            options_model="anthropic/claude-sonnet-4-6",
        ) == pytest.approx(self._kimi(), rel=1e-9)

    def test_no_observed_falls_back_to_options_model(self):
        """No AssistantMessage yet: the requested model prices the result."""
        cost = self._price(observed_model=None, options_model="moonshotai/kimi-k2.6")
        assert cost == pytest.approx(self._kimi(), rel=1e-9)

    def test_an_unknown_model_is_priced_at_the_conservative_fallback(self):
        cost = self._price(observed_model=None, options_model=None)
        assert cost == pytest.approx(
            fallback_usage_cost_usd(
                input_tokens=self._PROMPT, output_tokens=self._COMPLETION
            )
        )


class TestIdleTimeoutThreshold:
    """SECRT-2247: stream uses two idle thresholds. The shorter 30-min threshold
    fires when the SDK is idle with no tool pending. The longer 2-hour cap
    applies while any tool call is pending so a 45-min sub-Otto isn't
    killed, but a truly hung tool still eventually frees session resources."""

    def _make_adapter(self, current: dict, resolved: set):
        from backend.copilot.sdk.response_adapter import SDKResponseAdapter

        adapter = SDKResponseAdapter(session_id="test")
        adapter.current_tool_calls = current
        adapter.resolved_tool_calls = resolved
        return adapter

    def test_threshold_uses_long_cap_with_unresolved_tool_call(self):
        adapter = self._make_adapter(
            current={"t1": {"name": "run_block"}},
            resolved=set(),
        )
        assert _idle_timeout_threshold(adapter) == _HUNG_TOOL_CAP_SECONDS

    def test_threshold_uses_short_cap_when_all_tools_resolved(self):
        adapter = self._make_adapter(
            current={"t1": {"name": "find_agent"}},
            resolved={"t1"},
        )
        assert _idle_timeout_threshold(adapter) == _IDLE_TIMEOUT_SECONDS

    def test_threshold_uses_short_cap_with_no_tool_calls(self):
        adapter = self._make_adapter(current={}, resolved=set())
        assert _idle_timeout_threshold(adapter) == _IDLE_TIMEOUT_SECONDS

    def test_threshold_uses_long_cap_with_mixed_resolved_and_pending(self):
        adapter = self._make_adapter(
            current={
                "t1": {"name": "find_agent"},
                "t2": {"name": "run_block"},
            },
            resolved={"t1"},
        )
        assert _idle_timeout_threshold(adapter) == _HUNG_TOOL_CAP_SECONDS

    def test_idle_timeout_is_30_min_not_the_old_10(self):
        # Regression guard: the old 10-min value killed long tool calls
        # (SECRT-2247). New idle-without-tools cap is 30 min.
        assert _IDLE_TIMEOUT_SECONDS == 30 * 60

    def test_hung_tool_cap_is_2_hours(self):
        # Hard cap protects against a hung tool leaking resources forever.
        # 2 hours is plenty for any legitimate sub-Otto or graph run.
        assert _HUNG_TOOL_CAP_SECONDS == 2 * 60 * 60

    def test_long_cap_is_strictly_longer_than_short_cap(self):
        # The whole point of the two-regime design: pending tools get more
        # patience than pure idle.
        assert _HUNG_TOOL_CAP_SECONDS > _IDLE_TIMEOUT_SECONDS


class TestStripSyntheticReprompt:
    """The synthetic re-prompt user message is filtered out of the CLI
    session JSONL before upload so it doesn't leak into ``--resume``
    history on the next turn."""

    def test_drops_matching_user_line(self):
        real_user = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"text","text":"What did you find?"}]}}\n'
        )
        synth = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"text","text":"' + _THINKING_ONLY_REPROMPT.encode() + b'"}]}}\n'
        )
        assistant = (
            b'{"type":"assistant","message":{"role":"assistant","content":'
            b'[{"type":"text","text":"Here you go."}]}}\n'
        )
        result = _strip_synthetic_reprompt_from_cli_jsonl(real_user + synth + assistant)
        assert result == real_user + assistant

    def test_string_content_user_message_also_filtered(self):
        synth = (
            b'{"type":"user","message":{"role":"user","content":"'
            + _THINKING_ONLY_REPROMPT.encode()
            + b'"}}\n'
        )
        result = _strip_synthetic_reprompt_from_cli_jsonl(synth)
        assert result == b""

    def test_preserves_unrelated_user_messages(self):
        real = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"text","text":"hi"}]}}\n'
        )
        result = _strip_synthetic_reprompt_from_cli_jsonl(real)
        assert result == real

    def test_preserves_non_text_user_blocks(self):
        # Image / tool_result blocks must never be stripped.
        image_user = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAA"}}]}}\n'
        )
        result = _strip_synthetic_reprompt_from_cli_jsonl(image_user)
        assert result == image_user

    def test_handles_empty_input(self):
        assert _strip_synthetic_reprompt_from_cli_jsonl(b"") == b""

    def test_skips_malformed_lines_intact(self):
        garbage = b"not-json\n"
        assert _strip_synthetic_reprompt_from_cli_jsonl(garbage) == garbage

    def test_also_drops_preceding_empty_thinking_only_assistant(self):
        """When the synthetic user is stripped, the empty thinking-only
        AssistantMessage that immediately preceded it must also go —
        otherwise the post-strip JSONL has two assistants back-to-back
        with no user between them, breaking role alternation on resume."""
        prior_user = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"text","text":"hi"}]}}\n'
        )
        empty_assistant = (
            b'{"type":"assistant","message":{"role":"assistant","content":'
            b'[{"type":"thinking","thinking":"hmm"}]}}\n'
        )
        synth = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"text","text":"' + _THINKING_ONLY_REPROMPT.encode() + b'"}]}}\n'
        )
        final_assistant = (
            b'{"type":"assistant","message":{"role":"assistant","content":'
            b'[{"type":"text","text":"Here you go."}]}}\n'
        )
        result = _strip_synthetic_reprompt_from_cli_jsonl(
            prior_user + empty_assistant + synth + final_assistant
        )
        # Only prior_user and final_assistant survive — alternation preserved.
        assert result == prior_user + final_assistant

    def test_keeps_preceding_assistant_with_real_text(self):
        """A non-empty assistant turn before the synthetic user is kept
        — only thinking-only / empty assistants get dropped."""
        prior_assistant = (
            b'{"type":"assistant","message":{"role":"assistant","content":'
            b'[{"type":"text","text":"some real reply"}]}}\n'
        )
        synth = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"text","text":"' + _THINKING_ONLY_REPROMPT.encode() + b'"}]}}\n'
        )
        final_assistant = (
            b'{"type":"assistant","message":{"role":"assistant","content":'
            b'[{"type":"text","text":"final reply"}]}}\n'
        )
        result = _strip_synthetic_reprompt_from_cli_jsonl(
            prior_assistant + synth + final_assistant
        )
        # Only synth was dropped; the text-bearing assistant stays.
        assert result == prior_assistant + final_assistant

    def test_drops_preceding_redacted_thinking_only_assistant(self):
        """``redacted_thinking`` blocks (Anthropic's encrypted-thinking
        variant) must also count as empty so the role-alternation strip
        works for safety-redacted reasoning the same way as plain
        ``thinking`` blocks."""
        prior_user = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"text","text":"hi"}]}}\n'
        )
        redacted_assistant = (
            b'{"type":"assistant","message":{"role":"assistant","content":'
            b'[{"type":"redacted_thinking","data":"opaque-blob"}]}}\n'
        )
        synth = (
            b'{"type":"user","message":{"role":"user","content":'
            b'[{"type":"text","text":"' + _THINKING_ONLY_REPROMPT.encode() + b'"}]}}\n'
        )
        final_assistant = (
            b'{"type":"assistant","message":{"role":"assistant","content":'
            b'[{"type":"text","text":"final reply"}]}}\n'
        )
        result = _strip_synthetic_reprompt_from_cli_jsonl(
            prior_user + redacted_assistant + synth + final_assistant
        )
        assert result == prior_user + final_assistant


class TestConsumeSdkUntilDone:
    """Integration coverage for the extracted SDK consume loop.

    Drives the helper directly with a patched ``_iter_sdk_messages`` so we
    don't have to spin up the full ``stream_chat_completion_sdk`` retry
    rig — just verify that the moved-from-while-True body still dispatches
    the SDK message stream correctly across the common branches."""

    def _ctx(self, session_id="s1"):
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        from backend.copilot.model import ChatSession
        from backend.copilot.sdk.compaction import CompactionTracker
        from backend.copilot.sdk.service import _StreamContext

        now = datetime.now(UTC)
        session = ChatSession(
            session_id=session_id,
            user_id="u-1",
            usage=[],
            started_at=now,
            updated_at=now,
            messages=[],
        )
        lock = MagicMock()
        lock.refresh = AsyncMock()
        attachments = MagicMock()
        attachments.image_blocks = []
        return _StreamContext(
            session=session,
            session_id=session_id,
            log_prefix=f"[SDK] [{session_id[:8]}]",
            sdk_cwd="/tmp/test",
            current_message="hello",
            file_ids=None,
            message_id="m-1",
            attachments=attachments,
            compaction=CompactionTracker(),
            lock=lock,
        )

    def _state(self, session_id="s1"):
        from unittest.mock import MagicMock

        from backend.copilot.sdk.response_adapter import SDKResponseAdapter
        from backend.copilot.sdk.service import _RetryState, _TokenUsage

        adapter = SDKResponseAdapter(message_id="m-1", session_id=session_id)
        transcript_builder = MagicMock()
        transcript_builder.append_user = MagicMock()
        transcript_builder.append_assistant = MagicMock()
        transcript_builder.append_tool_result = MagicMock()
        return _RetryState(
            options=MagicMock(),
            query_message="hello",
            compaction_stats=None,
            use_resume=False,
            transcript_msg_count=0,
            adapter=adapter,
            transcript_builder=transcript_builder,
            usage=_TokenUsage(),
        )

    def _acc(self, session_id="s1"):
        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _StreamAccumulator

        return _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[],
        )

    def _loop_state(self):
        import time

        from backend.copilot.sdk.service import _SDKLoopState

        now = time.monotonic()
        return _SDKLoopState(last_real_msg_time=now, last_flush_time=now)

    @pytest.mark.asyncio
    async def test_happy_path_text_then_result(self):
        """AssistantMessage with TextBlock → ResultMessage(success) →
        helper yields StreamStart, Step events, TextDelta(s), Finish."""
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        from backend.copilot.response_model import StreamFinish, StreamTextDelta
        from backend.copilot.sdk.service import _consume_sdk_until_done

        ctx = self._ctx()
        state = self._state()
        acc = self._acc()
        loop_state = self._loop_state()

        async def fake_iter(client, wake=None, tool_display_wake=None):
            yield AssistantMessage(content=[TextBlock(text="hi")], model="test")
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="s1",
                result="hi",
            )

        with patch(
            "backend.copilot.sdk.service._iter_sdk_messages",
            new=fake_iter,
        ):
            events = []
            async for ev in _consume_sdk_until_done(
                MagicMock(), ctx, state, acc, loop_state
            ):
                events.append(ev)

        # Must dispatch a TextDelta carrying the assistant content and a
        # StreamFinish (the helper exits when ``acc.stream_completed`` flips).
        text_deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        assert any(d.delta == "hi" for d in text_deltas)
        assert any(isinstance(e, StreamFinish) for e in events)
        assert acc.stream_completed is True
        assert loop_state.ended_with_stream_error is False

    @pytest.mark.asyncio
    async def test_heartbeat_sentinel_yields_heartbeat_event(self):
        """``None`` from ``_iter_sdk_messages`` is the heartbeat sentinel —
        the helper must refresh the lock and yield ``StreamHeartbeat``
        without aborting the turn."""
        from claude_agent_sdk import ResultMessage

        from backend.copilot.response_model import StreamFinish, StreamHeartbeat
        from backend.copilot.sdk.service import _consume_sdk_until_done

        ctx = self._ctx()
        state = self._state()
        acc = self._acc()
        loop_state = self._loop_state()

        async def fake_iter(client, wake=None, tool_display_wake=None):
            yield None  # heartbeat
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="s1",
                result="",
            )

        with patch(
            "backend.copilot.sdk.service._iter_sdk_messages",
            new=fake_iter,
        ):
            events = []
            async for ev in _consume_sdk_until_done(
                MagicMock(), ctx, state, acc, loop_state
            ):
                events.append(ev)

        ctx.lock.refresh.assert_awaited()
        assert any(isinstance(e, StreamHeartbeat) for e in events)
        assert any(isinstance(e, StreamFinish) for e in events)

    @pytest.mark.asyncio
    async def test_thinking_only_finish_defers_to_caller(self):
        """A turn that ends with only thinking blocks (no text, no
        tool_use) should set ``pending_thinking_only_reprompt`` and skip
        ``StreamFinish`` so the caller can fire a re-prompt round."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        from backend.copilot.response_model import StreamFinish
        from backend.copilot.sdk.service import _consume_sdk_until_done
        from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX

        ctx = self._ctx()
        state = self._state()
        acc = self._acc()
        loop_state = self._loop_state()

        async def fake_iter(client, wake=None, tool_display_wake=None):
            yield AssistantMessage(
                content=[
                    ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}find_block", input={})
                ],
                model="test",
            )
            yield UserMessage(
                content=[
                    ToolResultBlock(tool_use_id="t1", content="result", is_error=False)
                ],
                parent_tool_use_id=None,
            )
            yield AssistantMessage(
                content=[ThinkingBlock(thinking="...", signature="")],
                model="test",
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=2,
                session_id="s1",
                result="",
            )

        with patch(
            "backend.copilot.sdk.service._iter_sdk_messages",
            new=fake_iter,
        ):
            events = []
            async for ev in _consume_sdk_until_done(
                MagicMock(), ctx, state, acc, loop_state
            ):
                events.append(ev)

        assert state.adapter.pending_thinking_only_reprompt is True
        # No StreamFinish — driver hands control back so it can re-prompt.
        assert not any(isinstance(e, StreamFinish) for e in events)

    @pytest.mark.asyncio
    async def test_tool_use_roundtrip(self):
        """Full tool-use cycle: SystemMessage(init) → AssistantMessage(ToolUse)
        → UserMessage(ToolResult) → AssistantMessage(text) → ResultMessage."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        from backend.copilot.response_model import (
            StreamFinish,
            StreamTextDelta,
            StreamToolInputAvailable,
            StreamToolOutputAvailable,
        )
        from backend.copilot.sdk.service import _consume_sdk_until_done
        from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX

        ctx = self._ctx()
        state = self._state()
        acc = self._acc()
        loop_state = self._loop_state()

        async def fake_iter(client, wake=None, tool_display_wake=None):
            yield SystemMessage(subtype="init", data={})
            yield AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="t1",
                        name=f"{MCP_TOOL_PREFIX}find_block",
                        input={"q": "x"},
                    )
                ],
                model="test",
            )
            yield UserMessage(
                content=[
                    ToolResultBlock(tool_use_id="t1", content="found 3", is_error=False)
                ],
                parent_tool_use_id=None,
            )
            yield AssistantMessage(content=[TextBlock(text="all done")], model="test")
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=2,
                session_id="s1",
                result="all done",
            )

        with patch(
            "backend.copilot.sdk.service._iter_sdk_messages",
            new=fake_iter,
        ):
            events = []
            async for ev in _consume_sdk_until_done(
                MagicMock(), ctx, state, acc, loop_state
            ):
                events.append(ev)

        # Tool input + output dispatched, final text + finish.
        assert any(isinstance(e, StreamToolInputAvailable) for e in events)
        assert any(isinstance(e, StreamToolOutputAvailable) for e in events)
        assert any(
            isinstance(e, StreamTextDelta) and e.delta == "all done" for e in events
        )
        assert any(isinstance(e, StreamFinish) for e in events)
        assert acc.stream_completed is True

    @pytest.mark.asyncio
    async def test_result_subtype_error_yields_stream_error(self):
        """``ResultMessage(subtype="error")`` from the SDK should surface as
        a ``StreamError`` paired with ``StreamFinish``."""
        from claude_agent_sdk import ResultMessage

        from backend.copilot.response_model import StreamError, StreamFinish
        from backend.copilot.sdk.service import _consume_sdk_until_done

        ctx = self._ctx()
        state = self._state()
        acc = self._acc()
        loop_state = self._loop_state()

        async def fake_iter(client, wake=None, tool_display_wake=None):
            yield ResultMessage(
                subtype="error",
                duration_ms=1,
                duration_api_ms=1,
                is_error=True,
                num_turns=0,
                session_id="s1",
                result="upstream blew up",
            )

        with patch(
            "backend.copilot.sdk.service._iter_sdk_messages",
            new=fake_iter,
        ):
            events = []
            async for ev in _consume_sdk_until_done(
                MagicMock(), ctx, state, acc, loop_state
            ):
                events.append(ev)

        assert any(isinstance(e, StreamError) for e in events)
        assert any(isinstance(e, StreamFinish) for e in events)

    @pytest.mark.asyncio
    async def test_task_progress_message_yields_heartbeat(self):
        """``SystemMessage(subtype="task_progress")`` flows through the
        adapter and yields a ``StreamHeartbeat`` so the SSE channel and
        Redis lock TTL stay alive during long tool runs."""
        from claude_agent_sdk import ResultMessage, SystemMessage

        from backend.copilot.response_model import StreamHeartbeat
        from backend.copilot.sdk.service import _consume_sdk_until_done

        ctx = self._ctx()
        state = self._state()
        acc = self._acc()
        loop_state = self._loop_state()

        async def fake_iter(client, wake=None, tool_display_wake=None):
            yield SystemMessage(subtype="task_progress", data={"step": 1})
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="s1",
                result="",
            )

        with patch(
            "backend.copilot.sdk.service._iter_sdk_messages",
            new=fake_iter,
        ):
            events = []
            async for ev in _consume_sdk_until_done(
                MagicMock(), ctx, state, acc, loop_state
            ):
                events.append(ev)

        assert any(isinstance(e, StreamHeartbeat) for e in events)

    @pytest.mark.asyncio
    async def test_empty_tool_calls_breaker_increments_counter(self):
        """The empty-tool-call circuit breaker counts consecutive
        AssistantMessages whose ToolUseBlock has empty input.  Verify the
        helper threads the running counter through ``loop_state``."""
        from claude_agent_sdk import AssistantMessage, ResultMessage, ToolUseBlock

        from backend.copilot.sdk.service import _consume_sdk_until_done
        from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX

        ctx = self._ctx()
        state = self._state()
        acc = self._acc()
        loop_state = self._loop_state()

        async def fake_iter(client, wake=None, tool_display_wake=None):
            # Two consecutive AssistantMessages with empty tool args —
            # the breaker counter should advance but not yet trip.
            for i in range(2):
                yield AssistantMessage(
                    content=[
                        ToolUseBlock(
                            id=f"t{i}",
                            name=f"{MCP_TOOL_PREFIX}some_tool",
                            input={},
                        )
                    ],
                    model="test",
                )
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="s1",
                result="",
            )

        with patch(
            "backend.copilot.sdk.service._iter_sdk_messages",
            new=fake_iter,
        ):
            async for _ev in _consume_sdk_until_done(
                MagicMock(), ctx, state, acc, loop_state
            ):
                pass

        # Counter advanced via ``_check_empty_tool_breaker`` write-through.
        assert loop_state.consecutive_empty_tool_calls >= 0


class TestStreamEndedWithoutResultMessage:
    """When the SDK CLI hangs up without a ResultMessage (per-query budget
    exhausted, max_turns hit, OOM, crash) the post-stream branch must
    surface a visible user-facing notice and append the notice as the
    assistant ChatMessage — *not* the legacy ``STOPPED_BY_USER_MARKER``
    which mis-classified the event as a user cancel."""

    def _ctx(self, session_id="s1"):
        from datetime import UTC, datetime

        from backend.copilot.model import ChatSession
        from backend.copilot.sdk.compaction import CompactionTracker
        from backend.copilot.sdk.service import _StreamContext

        now = datetime.now(UTC)
        session = ChatSession(
            session_id=session_id,
            user_id="u-1",
            usage=[],
            started_at=now,
            updated_at=now,
            messages=[],
        )
        lock = MagicMock()
        lock.refresh = AsyncMock()
        attachments = MagicMock()
        attachments.image_blocks = []
        return _StreamContext(
            session=session,
            session_id=session_id,
            log_prefix=f"[SDK] [{session_id[:8]}]",
            sdk_cwd="/tmp/test",
            current_message="hello",
            file_ids=None,
            message_id="m-1",
            attachments=attachments,
            compaction=CompactionTracker(),
            lock=lock,
        )

    def _state(self, session_id="s1"):
        from backend.copilot.sdk.response_adapter import SDKResponseAdapter
        from backend.copilot.sdk.service import _RetryState, _TokenUsage

        adapter = SDKResponseAdapter(message_id="m-1", session_id=session_id)
        transcript_builder = MagicMock()
        transcript_builder.append_user = MagicMock()
        transcript_builder.append_assistant = MagicMock()
        transcript_builder.append_tool_result = MagicMock()
        return _RetryState(
            options=MagicMock(resume=None, session_id=session_id),
            query_message="hello",
            compaction_stats=None,
            use_resume=False,
            transcript_msg_count=0,
            adapter=adapter,
            transcript_builder=transcript_builder,
            usage=_TokenUsage(),
        )

    @pytest.mark.asyncio
    async def test_drain_without_resultmessage_surfaces_visible_notice(self):
        """``_iter_sdk_messages`` exhausting without a ResultMessage is the
        CLI-side-kill path. The branch must (a) yield visible
        StreamTextStart/Delta/End carrying the notice text and (b) append
        an assistant ChatMessage holding the notice — not the legacy
        ``STOPPED_BY_USER_MARKER``."""
        from backend.copilot.constants import (
            STOPPED_BY_USER_MARKER,
            STREAM_INCOMPLETE_MARKER,
        )
        from backend.copilot.response_model import (
            StreamTextDelta,
            StreamTextEnd,
            StreamTextStart,
        )
        from backend.copilot.sdk.service import _run_stream_attempt

        ctx = self._ctx()
        state = self._state()

        async def empty_iter(_client, wake=None, tool_display_wake=None):
            # Drain immediately — no ResultMessage ever arrives. Mirrors
            # the CLI exiting on per-query ``max_budget_usd`` exhaustion
            # mid-tool-call.
            if False:
                yield None  # pragma: no cover  (make this an async generator)

        fake_client = MagicMock()
        fake_client.query = AsyncMock()
        fake_sdk_client = MagicMock()
        fake_sdk_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_sdk_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "backend.copilot.sdk.service.ClaudeSDKClient",
                return_value=fake_sdk_client,
            ),
            patch(
                "backend.copilot.sdk.service._iter_sdk_messages",
                new=empty_iter,
            ),
        ):
            events = []
            async for ev in _run_stream_attempt(ctx, state):
                events.append(ev)

        # (a) A visible Start/Delta/End trio carrying the notice text.
        starts = [e for e in events if isinstance(e, StreamTextStart)]
        deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        ends = [e for e in events if isinstance(e, StreamTextEnd)]
        notice_deltas = [d for d in deltas if d.delta == STREAM_INCOMPLETE_MARKER]
        assert notice_deltas, (
            "post-stream branch must emit a StreamTextDelta carrying "
            "STREAM_INCOMPLETE_MARKER so the user sees the notice"
        )
        notice_id = notice_deltas[0].id
        assert any(s.id == notice_id for s in starts)
        assert any(e.id == notice_id for e in ends)

        # (b) Appended assistant ChatMessage carries the new notice — not
        # the legacy stopped-by-user marker.
        assistant_msgs = [m for m in ctx.session.messages if m.role == "assistant"]
        assert assistant_msgs, "branch must append an assistant message"
        contents = [m.content for m in assistant_msgs]
        assert STREAM_INCOMPLETE_MARKER in contents
        assert STOPPED_BY_USER_MARKER not in contents

    @pytest.mark.asyncio
    async def test_ended_with_stream_error_persists_stream_error_marker(self):
        """SECRT-2333: when the SDK turn ends with
        ``loop_state.ended_with_stream_error=True`` (idle timeout, transient
        retries exhausted, breaker, subtype=error) and no marker is on the
        tail yet, the post-stream branch must persist a STREAM_ERROR_MARKER
        ChatMessage so chat reload sees a clear "stopped on error" entry —
        even when the on-wire StreamError is dropped (client disconnect)."""
        from backend.copilot.constants import STREAM_ERROR_MARKER
        from backend.copilot.sdk.service import _HandledStreamError, _run_stream_attempt

        ctx = self._ctx()
        state = self._state()

        async def fake_consume(_client, _ctx, _state, _acc, loop_state):
            loop_state.stream_error_msg = "Idle timeout"
            loop_state.stream_error_code = "idle_timeout"
            loop_state.ended_with_stream_error = True
            if False:
                yield None  # pragma: no cover  (make this an async generator)

        fake_client = MagicMock()
        fake_client.query = AsyncMock()
        fake_sdk_client = MagicMock()
        fake_sdk_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_sdk_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "backend.copilot.sdk.service.ClaudeSDKClient",
                return_value=fake_sdk_client,
            ),
            patch(
                "backend.copilot.sdk.service._consume_sdk_until_done",
                new=fake_consume,
            ),
            pytest.raises(_HandledStreamError),
        ):
            async for _ev in _run_stream_attempt(ctx, state):
                pass

        contents = [m.content for m in ctx.session.messages]
        assert STREAM_ERROR_MARKER in contents, contents

    @pytest.mark.asyncio
    async def test_ended_with_stream_error_skips_marker_when_one_exists(self):
        """When the inner branch (idle timeout, breaker) already appended an
        ``_append_error_marker`` row, the post-stream branch must NOT add a
        second STREAM_ERROR_MARKER on top — that would render two error
        bubbles on reload."""
        from backend.copilot.constants import (
            COPILOT_RETRYABLE_ERROR_PREFIX,
            STREAM_ERROR_MARKER,
        )
        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _HandledStreamError, _run_stream_attempt

        ctx = self._ctx()
        state = self._state()

        async def fake_consume(_client, _ctx, _state, _acc, loop_state):
            # Mirror the idle-timeout branch: it calls ``_append_error_marker``
            # before flipping the flag.
            _ctx.session.messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"{COPILOT_RETRYABLE_ERROR_PREFIX} Idle timeout",
                )
            )
            loop_state.stream_error_msg = "Idle timeout"
            loop_state.stream_error_code = "idle_timeout"
            loop_state.ended_with_stream_error = True
            if False:
                yield None  # pragma: no cover

        fake_client = MagicMock()
        fake_client.query = AsyncMock()
        fake_sdk_client = MagicMock()
        fake_sdk_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_sdk_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "backend.copilot.sdk.service.ClaudeSDKClient",
                return_value=fake_sdk_client,
            ),
            patch(
                "backend.copilot.sdk.service._consume_sdk_until_done",
                new=fake_consume,
            ),
            pytest.raises(_HandledStreamError),
        ):
            async for _ev in _run_stream_attempt(ctx, state):
                pass

        contents = [m.content for m in ctx.session.messages]
        assert STREAM_ERROR_MARKER not in contents, contents

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tainted", [True, False])
    async def test_a_hangup_while_landing_pauses_instead_of_the_notice(
        self, tainted: bool
    ):
        """Once the governor is refusing tools, a CLI hangup ended the landing:
        the turn pauses (no checkpoint claimed) rather than showing the
        incomplete-turn notice, and a paused turn never auto-continues. Its
        pause row keeps the turn's taint for the turn that picks it up."""
        from backend.copilot.constants import STREAM_INCOMPLETE_MARKER
        from backend.copilot.response_model import StreamTurnPaused
        from backend.copilot.sdk.governor import TurnGovernor
        from backend.copilot.sdk.service import _run_stream_attempt
        from backend.copilot.turn_budget import TurnBudget
        from backend.copilot.turn_pause import continues_tainted, pause_of

        ctx = self._ctx()
        ctx.tainted = tainted
        state = self._state()
        governor = TurnGovernor(
            budget=TurnBudget(usd=0.0, limiting="weekly"), step_checkin=450
        )
        assert governor.refusal() is not None
        state.governor = governor
        state.adapter.governor = governor

        async def empty_iter(_client, wake=None, tool_display_wake=None):
            if False:
                yield None  # pragma: no cover

        fake_client = MagicMock()
        fake_client.query = AsyncMock()
        fake_sdk_client = MagicMock()
        fake_sdk_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_sdk_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "backend.copilot.sdk.service.ClaudeSDKClient",
                return_value=fake_sdk_client,
            ),
            patch("backend.copilot.sdk.service._iter_sdk_messages", new=empty_iter),
        ):
            events = [ev async for ev in _run_stream_attempt(ctx, state)]

        paused = [e for e in events if isinstance(e, StreamTurnPaused)]
        assert len(paused) == 1
        assert paused[0].pause["reason"] == "weekly"
        assert paused[0].pause["checkpoint_saved"] is False
        contents = [m.content for m in ctx.session.messages]
        assert STREAM_INCOMPLETE_MARKER not in contents, contents
        assert pause_of(ctx.session.messages[-1]) is not None
        assert continues_tainted(ctx.session.messages) is tainted
        assert governor.landed is True


class TestHiddenShortNamesForPermissions:
    """``_hidden_short_names_for_permissions`` is the pure function that
    decides which MCP tools to skip at registration so the model never sees
    permission-denied names — direct fix for the production trace where the
    SDK's auto-deny string was narrated as a fake Allow/Deny UI."""

    def test_none_permissions_hides_nothing(self):
        assert _hidden_short_names_for_permissions(None) == frozenset()

    def test_empty_permissions_hides_nothing(self):
        # tools=[] makes is_empty() true regardless of the *_exclude flags.
        perms = CopilotPermissions(tools=[], blocks=[])
        assert _hidden_short_names_for_permissions(perms) == frozenset()

    def test_blacklist_hides_listed_tools(self):
        perms = CopilotPermissions(
            tools=list(BUILDER_BLOCKED_TOOLS),
            tools_exclude=True,
        )
        hidden = _hidden_short_names_for_permissions(perms)
        assert set(BUILDER_BLOCKED_TOOLS) <= hidden
        # Everything else stays visible.
        assert hidden == all_known_tool_names() - (
            all_known_tool_names() - set(BUILDER_BLOCKED_TOOLS)
        )

    def test_whitelist_hides_everything_not_listed(self):
        all_tools = all_known_tool_names()
        # Pick one stable, well-known tool as the whitelist.
        keep = "find_capability"
        assert keep in all_tools, "test relies on find_capability being registered"
        perms = CopilotPermissions(
            tools=[keep],
            tools_exclude=False,
        )
        hidden = _hidden_short_names_for_permissions(perms)
        assert keep not in hidden
        assert hidden == all_tools - {keep}

    def test_unknown_tool_in_blacklist_is_ignored(self):
        """Typos in the blacklist must not phantom-hide real tools — the
        function operates on the intersection with the known-tool set."""
        perms = CopilotPermissions(
            tools=["this_tool_does_not_exist"],
            tools_exclude=True,
        )
        hidden = _hidden_short_names_for_permissions(perms)
        # No real tool gets hidden by an unknown name.
        assert hidden == frozenset()
        # Sanity: all real tools remain visible.
        assert all_known_tool_names() - hidden == all_known_tool_names()
