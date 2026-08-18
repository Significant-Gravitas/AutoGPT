"""Tests for backend/copilot/local_context_probe.py.

Covers the core compaction-target regression (no more 120k default for local
models) plus all four probe strategies and the fallback/cache paths. All probes
are mocked — no network.
"""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.local_context_probe import (
    _MINIMUM_SAFE_WINDOW,
    LOCAL_CONTEXT_FALLBACK,
    _last_window,
    _probe_cache,
    _server_root,
    compaction_target_for_window,
    probe_local_context_window,
)


def _resp(status: int, body: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    return r


def _mock_client(responses) -> MagicMock:
    """An httpx.AsyncClient mock whose GETs yield *responses* in order.

    *responses* may be a list (one per GET) or a single Exception raised on
    every GET.
    """
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(side_effect=responses)
    return client


class TestCompactionTargetForWindow:
    def test_32k_window_is_not_the_broken_120k_default(self):
        # The whole point: a 32k local window must yield ~9k, not 120_000.
        assert compaction_target_for_window(32_768) == 8_768

    def test_64k_window(self):
        assert compaction_target_for_window(65_536) == 41_536

    def test_tiny_window_clamps_to_floor(self):
        assert compaction_target_for_window(4_096) == 4_096
        assert compaction_target_for_window(1_000) == 4_096

    def test_fallback_constant_yields_sane_target(self):
        assert compaction_target_for_window(LOCAL_CONTEXT_FALLBACK) == 8_768


class TestServerRoot:
    def test_strips_v1(self):
        assert _server_root("http://host:11434/v1") == "http://host:11434"

    def test_strips_v1_trailing_slash(self):
        assert _server_root("http://host:11434/v1/") == "http://host:11434"

    def test_no_v1(self):
        assert _server_root("http://host:11434") == "http://host:11434"

    def test_https(self):
        assert (
            _server_root("https://ollama.example.com/v1")
            == "https://ollama.example.com"
        )


class TestProbeStrategies:
    def setup_method(self):
        _probe_cache.clear()
        _last_window.clear()

    @pytest.mark.asyncio
    async def test_ollama_api_ps_fixes_compaction_bug(self):
        client = _mock_client(
            [
                _resp(
                    200,
                    {
                        "models": [
                            {
                                "name": "llama3.1:8b-instruct-q4_K_M",
                                "context_length": 32_768,
                            }
                        ]
                    },
                )
            ]
        )
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ):
            window = await probe_local_context_window(
                "http://localhost:11434/v1", "llama3.1:8b-instruct-q4_K_M"
            )
        assert window == 32_768
        assert compaction_target_for_window(window) == 8_768

    @pytest.mark.asyncio
    async def test_ollama_falls_back_to_first_loaded_model_window(self):
        # Name doesn't match, but a model is loaded — window is server-wide.
        client = _mock_client(
            [_resp(200, {"models": [{"name": "other:7b", "context_length": 16_384}]})]
        )
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ):
            window = await probe_local_context_window(
                "http://localhost:11434/v1", "mymodel:8b"
            )
        assert window == 16_384

    @pytest.mark.asyncio
    async def test_empty_ollama_falls_through_to_vllm(self):
        client = _mock_client(
            [
                _resp(200, {"models": []}),  # /api/ps: nothing loaded
                _resp(404, {}),  # /props: not llama.cpp
                _resp(
                    200, {"data": [{"id": "llama3.1:8b", "max_model_len": 32_768}]}
                ),  # vLLM
            ]
        )
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ):
            window = await probe_local_context_window(
                "http://localhost:11434/v1", "llama3.1:8b"
            )
        assert window == 32_768

    @pytest.mark.asyncio
    async def test_llamacpp_props(self):
        client = _mock_client(
            [
                _resp(404, {}),  # /api/ps
                _resp(200, {"default_generation_settings": {"n_ctx": 8192}}),  # /props
            ]
        )
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ):
            window = await probe_local_context_window(
                "http://localhost:8080/v1", "llama-3-8b.gguf"
            )
        assert window == 8192

    @pytest.mark.asyncio
    async def test_lmstudio_api_v0(self):
        client = _mock_client(
            [
                _resp(404, {}),  # /api/ps
                _resp(404, {}),  # /props
                _resp(200, {"data": []}),  # /v1/models: no max_model_len
                _resp(
                    200, {"data": [{"id": "llama-3-8b", "max_context_length": 32_768}]}
                ),  # LM Studio
            ]
        )
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ):
            window = await probe_local_context_window(
                "http://localhost:1234/v1", "llama-3-8b"
            )
        assert window == 32_768


class TestFallbackAndWarnings:
    def setup_method(self):
        _probe_cache.clear()
        _last_window.clear()

    @pytest.mark.asyncio
    async def test_all_probes_fail_returns_fallback(self):
        client = _mock_client(OSError("connection refused"))
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ):
            window = await probe_local_context_window(
                "http://localhost:11434/v1", "llama3.1:8b"
            )
        assert window == LOCAL_CONTEXT_FALLBACK
        assert compaction_target_for_window(window) > 0

    @pytest.mark.asyncio
    async def test_window_below_minimum_logs_warning(self, caplog):
        client = _mock_client(
            [_resp(200, {"models": [{"name": "tiny", "context_length": 8192}]})]
        )
        with (
            patch(
                "backend.copilot.local_context_probe.httpx.AsyncClient",
                return_value=client,
            ),
            caplog.at_level(
                logging.WARNING, logger="backend.copilot.local_context_probe"
            ),
        ):
            window = await probe_local_context_window(
                "http://localhost:11434/v1", "tiny"
            )
        assert window == 8192
        assert window < _MINIMUM_SAFE_WINDOW
        assert any("OLLAMA_CONTEXT_LENGTH" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_probe_miss_reuses_last_known_window(self):
        """After a successful read, a later miss (model unloaded) reuses the
        last detected window — not the optimistic 32k constant."""
        client = _mock_client(
            [
                _resp(200, {"models": [{"name": "m", "context_length": 8192}]}),
                _resp(200, {"models": []}),  # 2nd probe: model unloaded
                _resp(404, {}),
                _resp(404, {}),
                _resp(404, {}),
            ]
        )
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ):
            w1 = await probe_local_context_window("http://h:11434/v1", "m")
            _probe_cache[("http://h:11434/v1", "m")] = (
                w1,
                time.monotonic() - 400,
            )  # force-expire so the next call re-probes
            w2 = await probe_local_context_window("http://h:11434/v1", "m")
        assert w1 == 8192
        assert w2 == 8192  # reused, not LOCAL_CONTEXT_FALLBACK
        assert w2 != LOCAL_CONTEXT_FALLBACK


class TestCaching:
    def setup_method(self):
        _probe_cache.clear()
        _last_window.clear()

    @pytest.mark.asyncio
    async def test_cache_hit_skips_probe(self):
        client = _mock_client(
            [_resp(200, {"models": [{"name": "m", "context_length": 32_768}]})]
        )
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ) as cls:
            w1 = await probe_local_context_window("http://localhost:11434/v1", "m")
            w2 = await probe_local_context_window("http://localhost:11434/v1", "m")
        assert w1 == w2 == 32_768
        assert cls.call_count == 1  # AsyncClient instantiated only once

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self):
        client = _mock_client(
            [
                _resp(200, {"models": [{"name": "m", "context_length": 32_768}]}),
                _resp(200, {"models": [{"name": "m", "context_length": 65_536}]}),
            ]
        )
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ) as cls:
            w1 = await probe_local_context_window("http://h:11434/v1", "m")
            _probe_cache[("http://h:11434/v1", "m")] = (
                w1,
                time.monotonic() - 400,
            )  # force-expire
            w2 = await probe_local_context_window("http://h:11434/v1", "m")
        assert w1 == 32_768
        assert w2 == 65_536
        assert cls.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_keyed_by_base_url_and_model(self):
        """Two models on the same endpoint don't share a cached window —
        vLLM / LM Studio expose a per-model window."""
        loaded = {
            "models": [
                {"name": "modelA", "context_length": 32_768},
                {"name": "modelB", "context_length": 16_384},
            ]
        }
        client = _mock_client([_resp(200, loaded), _resp(200, loaded)])
        with patch(
            "backend.copilot.local_context_probe.httpx.AsyncClient", return_value=client
        ) as cls:
            wa = await probe_local_context_window("http://h:11434/v1", "modelA")
            wb = await probe_local_context_window("http://h:11434/v1", "modelB")
        assert wa == 32_768
        assert wb == 16_384  # not modelA's cached 32_768
        assert cls.call_count == 2  # second model re-probed, not served from cache
