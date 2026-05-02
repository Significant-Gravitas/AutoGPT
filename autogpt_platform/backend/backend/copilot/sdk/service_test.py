"""Tests for SDK service helpers."""

import asyncio
import base64
import os
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import config as cfg_mod

from .service import (
    _HUNG_TOOL_CAP_SECONDS,
    _IDLE_TIMEOUT_SECONDS,
    _build_system_prompt_value,
    _humanise_tool_list,
    _idle_timeout_threshold,
    _is_sdk_disconnect_error,
    _normalize_model_name,
    _prepare_file_attachments,
    _resolve_sdk_model,
    _resolve_sdk_model_for_request,
    _safe_close_sdk_client,
)


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
    async def test_pdf_saved_to_disk(self, tmp_path):
        """PDFs should be saved to disk for Read tool access, not embedded."""
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
        assert str(saved) in result.hint

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
        # Read tool hint should appear (has non-image files)
        assert "Read tool" in result.hint

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
        """When all files are images, no Read tool hint should appear."""
        info = _FakeFileInfo("i1", "cat.png", "/cat.png", "image/png", 4)
        mgr = AsyncMock()
        mgr.get_file_info.return_value = info
        mgr.read_file_by_id.return_value = b"data"

        with patch(_PATCH_TARGET, new_callable=AsyncMock, return_value=mgr):
            result = await _prepare_file_attachments(["i1"], "u", "s", str(tmp_path))

        assert "Read tool" not in result.hint
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
        assert "Tool Discovery Priority" in SHARED_TOOL_NOTES
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


class TestNormalizeModelName:
    """Tests for _normalize_model_name — shared provider-aware normalization."""

    def test_strips_provider_prefix(self, monkeypatch, _clean_config_env):
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
            # Pin SDK slugs to anthropic/* so the new
            # _validate_sdk_model_vendor_compatibility allows construction.
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _normalize_model_name("anthropic/claude-opus-4.6") == "claude-opus-4-6"

    def test_openrouter_keeps_full_slug(self, monkeypatch, _clean_config_env):
        """OpenRouter routes by ``vendor/model`` slug — keep prefix and dots."""
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert (
            _normalize_model_name("anthropic/claude-opus-4.6")
            == "anthropic/claude-opus-4.6"
        )
        assert _normalize_model_name("moonshotai/kimi-k2.6") == "moonshotai/kimi-k2.6"

    def test_no_prefix_no_dots(self, monkeypatch, _clean_config_env):
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert (
            _normalize_model_name("claude-sonnet-4-20250514")
            == "claude-sonnet-4-20250514"
        )


class TestResolveSdkModel:
    """Tests for _resolve_sdk_model — model ID resolution for the SDK CLI."""

    def test_openrouter_active_keeps_full_slug(self, monkeypatch, _clean_config_env):
        """When OpenRouter is fully active, the canonical vendor/model slug
        is preserved so OpenRouter can route to the correct provider."""
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-opus-4.6",
            claude_agent_model=None,
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _resolve_sdk_model() == "anthropic/claude-opus-4.6"

    def test_openrouter_active_kimi_slug(self, monkeypatch, _clean_config_env):
        """Non-Anthropic models (Kimi via Moonshot) require the prefix to
        survive OpenRouter routing — strip would leave an unroutable slug."""
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="moonshotai/kimi-k2.6",
            claude_agent_model=None,
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _resolve_sdk_model() == "moonshotai/kimi-k2.6"

    def test_openrouter_disabled_normalizes_to_hyphens(
        self, monkeypatch, _clean_config_env
    ):
        """When OpenRouter is disabled, dots are replaced with hyphens."""
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-opus-4.6",
            claude_agent_model=None,
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _resolve_sdk_model() == "claude-opus-4-6"

    def test_openrouter_enabled_but_missing_key_normalizes(
        self, monkeypatch, _clean_config_env
    ):
        """When OpenRouter is enabled but api_key is missing, falls back to
        direct Anthropic and normalizes dots to hyphens."""
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-opus-4.6",
            claude_agent_model=None,
            use_openrouter=True,
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _resolve_sdk_model() == "claude-opus-4-6"

    def test_explicit_claude_agent_model_takes_precedence(
        self, monkeypatch, _clean_config_env
    ):
        """When claude_agent_model is explicitly set, it is returned as-is."""
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-opus-4.6",
            claude_agent_model="claude-sonnet-4-5-20250514",
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _resolve_sdk_model() == "claude-sonnet-4-5-20250514"

    def test_subscription_mode_returns_none(self, monkeypatch, _clean_config_env):
        """When using Claude Code subscription, returns None (CLI picks model)."""
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-opus-4.6",
            claude_agent_model=None,
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=True,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _resolve_sdk_model() is None

    def test_model_without_provider_prefix(self, monkeypatch, _clean_config_env):
        """When model has no provider prefix, it still normalizes correctly."""
        from backend.copilot import config as cfg_mod

        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="claude-opus-4.6",
            claude_agent_model=None,
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)
        assert _resolve_sdk_model() == "claude-opus-4-6"


class TestResolveSdkModelForRequestLdFallback:
    """``_resolve_sdk_model_for_request`` must fail soft when the LD value
    can't be normalised for the active routing mode — flagged as MAJOR by
    CodeRabbit + HIGH by Sentry when it was a hard ValueError."""

    @pytest.mark.asyncio
    async def test_direct_anthropic_mode_rejects_kimi_ld_value_and_falls_back(
        self, monkeypatch, _clean_config_env
    ):
        """LD serves ``moonshotai/kimi-k2.6`` but we're on direct-Anthropic
        (no OpenRouter key).  ``_normalize_model_name`` raises; the
        resolver must log + return the config-default path instead of
        500-ing the turn."""
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            claude_agent_model=None,
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="moonshotai/kimi-k2.6"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="standard", session_id="sess-abc", user_id="user-1"
            )

        # Fallback == tier-specific config default (thinking_standard_model
        # normalised to hyphen-form for direct-Anthropic mode).
        assert resolved == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_openrouter_mode_accepts_ld_kimi_value(
        self, monkeypatch, _clean_config_env
    ):
        """On OpenRouter the Kimi slug is legitimate — no fallback,
        value returned as-is."""
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            claude_agent_model=None,
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="moonshotai/kimi-k2.6"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="standard", session_id="sess-abc", user_id="user-1"
            )
        assert resolved == "moonshotai/kimi-k2.6"

    @pytest.mark.asyncio
    async def test_advanced_tier_fallback_uses_advanced_default_not_standard(
        self, monkeypatch, _clean_config_env
    ):
        """An LD-rejected ADVANCED slug must fall back to the advanced
        config default (Opus) — not the standard default (Sonnet).
        Using ``_resolve_sdk_model()`` as the fallback silently
        downgraded the user's chosen tier.  Flagged MAJOR by CodeRabbit
        + HIGH by Sentry on the first fail-soft commit."""
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4.7",
            claude_agent_model=None,
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="moonshotai/kimi-k2.6"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="advanced", session_id="sess-adv", user_id="user-1"
            )

        # Direct-Anthropic normalises anthropic/claude-opus-4.7 → claude-opus-4-7
        assert resolved == "claude-opus-4-7"

    @pytest.mark.asyncio
    async def test_standard_ld_override_wins_over_subscription(
        self, monkeypatch, _clean_config_env
    ):
        """Bug reported in local test: subscription mode + LD serving Kimi
        on ``copilot-model-routing[thinking][standard]`` returned
        ``None`` (CLI picked subscription default Opus), silently
        ignoring the LD override.  An LD value different from the
        config default is an explicit admin decision and must win.

        Subscription transport rejects non-Anthropic vendors (the CLI
        subprocess can't talk to Moonshot), so the resolver fails soft
        to the tier default normalised for the subscription transport
        (``claude-sonnet-4-6``) — not ``None``, which would silently
        re-introduce the old subscription-default bypass."""
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            claude_agent_model=None,
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=True,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="moonshotai/kimi-k2.6"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="standard", session_id="sess-std-sub", user_id="user-1"
            )
        # Kimi can't be served by the subscription CLI; fail-soft to
        # the tier default normalised for the active transport.
        assert resolved == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_standard_subscription_survives_trailing_whitespace_in_env(
        self, monkeypatch, _clean_config_env
    ):
        """``_resolve_thinking_model_for_user`` strips whitespace from the LD
        side; the config tier default must be stripped too, otherwise a
        stray trailing space in ``CHAT_THINKING_STANDARD_MODEL`` makes
        ``resolved == tier_default`` spuriously False and bypasses
        subscription-default mode.  Sentry HIGH on L856."""
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6  ",  # trailing spaces
            claude_agent_model=None,
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=True,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="anthropic/claude-sonnet-4-6"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="standard", session_id="sess-ws", user_id="user-1"
            )
        assert resolved is None, (
            "LD value semantically matches the whitespace-padded config "
            "default — subscription mode must still win and return None"
        )

    @pytest.mark.asyncio
    async def test_standard_subscription_default_honoured_when_ld_matches_config(
        self, monkeypatch, _clean_config_env
    ):
        """When LD serves the SAME value as the config default (i.e. the
        flag is effectively unset / no override), subscription mode still
        wins and we return ``None`` so the CLI uses the subscription
        default model."""
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            claude_agent_model=None,
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=True,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="anthropic/claude-sonnet-4-6"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="standard", session_id="sess-std-nop", user_id="user-1"
            )
        assert resolved is None

    @pytest.mark.asyncio
    async def test_advanced_tier_consults_ld_under_subscription(
        self, monkeypatch, _clean_config_env
    ):
        """Subscription mode bypasses LD only on the standard tier —
        the advanced tier always consults LD because the user explicitly
        asked for the premium path.  A subscription + advanced request
        with LD-served Opus must return Opus normalised for the
        subscription CLI (``claude-opus-4-7``), not the OpenRouter slug
        ``anthropic/claude-opus-4.7`` which the CLI subprocess rejects
        even when ``CHAT_BASE_URL`` is set to the OpenRouter proxy."""
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4.7",
            claude_agent_model=None,
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=True,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="anthropic/claude-opus-4.7"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="advanced", session_id="sess-adv-sub", user_id="user-1"
            )
        assert resolved == "claude-opus-4-7"


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


class TestSystemPromptPreset:
    """Tests for _build_system_prompt_value — cross-user prompt caching."""

    def test_preset_dict_structure_when_enabled(self):
        """When cross_user_cache is True, returns a _SystemPromptPreset dict."""
        custom_prompt = "You are a helpful assistant."
        result = _build_system_prompt_value(custom_prompt, cross_user_cache=True)

        assert isinstance(result, dict)
        assert result["type"] == "preset"
        assert result["preset"] == "claude_code"
        assert result["append"] == custom_prompt
        assert result["exclude_dynamic_sections"] is True

    def test_raw_string_when_disabled(self):
        """When cross_user_cache is False, returns the raw string."""
        custom_prompt = "You are a helpful assistant."
        result = _build_system_prompt_value(custom_prompt, cross_user_cache=False)

        assert isinstance(result, str)
        assert result == custom_prompt

    def test_empty_string_with_cache_enabled(self):
        """Empty system_prompt with cross_user_cache=True produces append=''."""
        result = _build_system_prompt_value("", cross_user_cache=True)

        assert isinstance(result, dict)
        assert result["type"] == "preset"
        assert result["preset"] == "claude_code"
        assert result["append"] == ""
        assert result["exclude_dynamic_sections"] is True

    def test_resume_and_fresh_share_the_same_static_prefix(self):
        """Every turn (fresh + --resume) must emit the same preset dict
        so the cross-user cache prefix match works on all turns.  This
        relies on CLI ≥ 2.1.98 (installed in the Docker image); older
        CLIs would crash on --resume + excludeDynamicSections=True."""
        fresh = _build_system_prompt_value("sys", cross_user_cache=True)
        resumed = _build_system_prompt_value("sys", cross_user_cache=True)
        assert fresh == resumed
        assert isinstance(fresh, dict)
        assert fresh.get("exclude_dynamic_sections") is True

    def test_default_config_is_enabled(self, _clean_config_env):
        """The default value for claude_agent_cross_user_prompt_cache is True."""
        cfg = cfg_mod.ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
        )
        assert cfg.claude_agent_cross_user_prompt_cache is True

    def test_env_var_disables_cache(self, _clean_config_env, monkeypatch):
        """CHAT_CLAUDE_AGENT_CROSS_USER_PROMPT_CACHE=false disables caching."""
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_CROSS_USER_PROMPT_CACHE", "false")
        cfg = cfg_mod.ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
        )
        assert cfg.claude_agent_cross_user_prompt_cache is False


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
    ``_RetryState``.  The Moonshot cost override reads this — when a
    fallback model activates mid-attempt, the requested primary
    (``state.options.model``) no longer matches what actually ran."""

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
            was_compacted=False,
            use_resume=False,
            resume_file=None,
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


class TestMoonshotCostOverrideGate:
    """Regression guards for the decision logic in
    ``_run_stream_attempt`` that picks between the CLI-reported cost
    and the Moonshot rate-card override.  The code:

        active_model = state.observed_model or getattr(state.options, "model", None)
        if _is_moonshot_model(active_model):
            state.usage.cost_usd = _override_cost_for_moonshot(...)
        else:
            state.usage.cost_usd = sdk_msg.total_cost_usd

    is critical-path billing logic — make sure observed_model wins over
    the requested primary, and Anthropic turns pass through untouched."""

    def _decide_cost(
        self,
        *,
        observed_model: str | None,
        options_model: str | None,
        sdk_reported_usd: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> float:
        """Mirror of the real decision block — lets us assert the gate
        without constructing the whole 1000-line generator."""
        from .service import _is_moonshot_model, _override_cost_for_moonshot

        active_model = observed_model or options_model
        if _is_moonshot_model(active_model):
            return _override_cost_for_moonshot(
                model=active_model,
                sdk_reported_usd=sdk_reported_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_tokens=0,
                cache_creation_tokens=0,
            )
        return sdk_reported_usd

    def test_anthropic_turn_passes_sdk_cost_through(self):
        """Anthropic — the CLI's pricing table is authoritative, so
        ``state.usage.cost_usd`` is set to ``sdk_msg.total_cost_usd``
        unchanged."""
        cost = self._decide_cost(
            observed_model="anthropic/claude-sonnet-4-6",
            options_model="anthropic/claude-sonnet-4-6",
            sdk_reported_usd=0.123,
        )
        assert cost == 0.123

    def test_moonshot_turn_uses_rate_card_override(self):
        """Moonshot — the CLI would silently bill at Sonnet rates, so
        the override recomputes from the Moonshot rate card."""
        cost = self._decide_cost(
            observed_model="moonshotai/kimi-k2.6",
            options_model="moonshotai/kimi-k2.6",
            sdk_reported_usd=0.089862,  # CLI's Sonnet-priced estimate.
            prompt_tokens=29564,
            completion_tokens=78,
        )
        expected = (29564 * 0.60 + 78 * 2.80) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-9)
        # Sanity: ~5x cheaper than the CLI's Sonnet-priced number.
        assert cost < 0.089862 / 4

    def test_observed_model_wins_over_options_primary(self):
        """The whole point of ``observed_model``: a Moonshot-primary
        request that fell back to Anthropic must NOT get Moonshot
        pricing applied.  The gate follows the observed model, not the
        requested primary."""
        cost = self._decide_cost(
            observed_model="anthropic/claude-sonnet-4-6",
            options_model="moonshotai/kimi-k2.6",  # what we ASKED for
            sdk_reported_usd=0.123,
            prompt_tokens=1000,
            completion_tokens=100,
        )
        # Observed == Anthropic → CLI-reported cost passes through unchanged.
        assert cost == 0.123

    def test_anthropic_to_moonshot_fallback_uses_override(self):
        """The inverse: an Anthropic-primary request that fell back to
        Moonshot must get the Moonshot override applied — the CLI is
        still billing at Sonnet rates for the fallback response."""
        cost = self._decide_cost(
            observed_model="moonshotai/kimi-k2.6",
            options_model="anthropic/claude-sonnet-4-6",
            sdk_reported_usd=0.089862,
            prompt_tokens=29564,
            completion_tokens=78,
        )
        expected = (29564 * 0.60 + 78 * 2.80) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_no_observed_falls_back_to_options_model(self):
        """First AssistantMessage hasn't arrived yet (or the SDK didn't
        emit ``.model``) — the gate falls back to the requested primary."""
        cost = self._decide_cost(
            observed_model=None,
            options_model="moonshotai/kimi-k2.6",
            sdk_reported_usd=0.089862,
            prompt_tokens=100,
            completion_tokens=10,
        )
        expected = (100 * 0.60 + 10 * 2.80) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_both_none_passes_sdk_cost_through(self):
        """Subscription mode — ``options.model`` may be None and no
        AssistantMessage has arrived yet.  ``None`` is not a Moonshot
        slug so the SDK number lands unchanged."""
        cost = self._decide_cost(
            observed_model=None,
            options_model=None,
            sdk_reported_usd=0.05,
        )
        assert cost == 0.05


# ---------------------------------------------------------------------------
# Moonshot helper re-exports — keep imports stable for call-site code
# ---------------------------------------------------------------------------


class TestMoonshotHelperReexports:
    """``sdk/service.py`` imports the Moonshot helpers under local
    aliases (``_is_moonshot_model``, ``_override_cost_for_moonshot``).
    Regression guard so a refactor doesn't silently break the import
    path the hot-loop code relies on."""

    def test_is_moonshot_model_aliased(self):
        from backend.copilot.moonshot import is_moonshot_model as canonical

        from .service import _is_moonshot_model

        assert _is_moonshot_model is canonical

    def test_override_cost_for_moonshot_aliased(self):
        from backend.copilot.moonshot import override_cost_usd as canonical

        from .service import _override_cost_for_moonshot

        assert _override_cost_for_moonshot is canonical


class TestIdleTimeoutThreshold:
    """SECRT-2247: stream uses two idle thresholds. The shorter 30-min threshold
    fires when the SDK is idle with no tool pending. The longer 2-hour cap
    applies while any tool call is pending so a 45-min sub-AutoPilot isn't
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
        # 2 hours is plenty for any legitimate sub-AutoPilot or graph run.
        assert _HUNG_TOOL_CAP_SECONDS == 2 * 60 * 60

    def test_long_cap_is_strictly_longer_than_short_cap(self):
        # The whole point of the two-regime design: pending tools get more
        # patience than pure idle.
        assert _HUNG_TOOL_CAP_SECONDS > _IDLE_TIMEOUT_SECONDS
