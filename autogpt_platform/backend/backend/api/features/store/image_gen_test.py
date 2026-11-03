import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

from backend.api.features.store import image_gen
from backend.blocks.ideogram import IdeogramModelBlock


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


@pytest.fixture
def graph() -> SimpleNamespace:
    return SimpleNamespace(
        name="Customer Follow-up",
        description="Drafts a helpful response and routes it for review.",
    )


def _provider_jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncMock, Mock, AsyncMock]:
    """Both providers stubbed at their boundary, answering a JPEG, plus the
    daily cap's counter. Returns the Ideogram and Replicate calls and the
    counter."""
    ideogram = AsyncMock(return_value="https://ideogram.example/image.jpg")
    replicate = Mock(return_value="https://replicate.example/image.jpg")
    monkeypatch.setattr(IdeogramModelBlock, "run_once", ideogram)
    monkeypatch.setattr(
        image_gen, "ReplicateClient", Mock(return_value=Mock(run=replicate))
    )
    download = AsyncMock(return_value=httpx.Response(200, content=_provider_jpeg()))
    monkeypatch.setattr(image_gen, "Requests", Mock(return_value=Mock(get=download)))
    counter = AsyncMock(return_value=1)
    monkeypatch.setattr(image_gen, "get_redis_async", AsyncMock())
    monkeypatch.setattr(image_gen, "incr_with_ttl", counter)
    return ideogram, replicate, counter


def _configure_providers(
    monkeypatch: pytest.MonkeyPatch, use_v2: bool, api_key: str
) -> None:
    monkeypatch.setattr(
        image_gen.settings.config, "use_agent_image_generation_v2", use_v2
    )
    monkeypatch.setattr(image_gen.settings.secrets, "replicate_api_key", api_key)
    monkeypatch.setattr(image_gen.ideogram_credentials, "api_key", SecretStr(api_key))


@pytest.mark.asyncio
@pytest.mark.parametrize("use_v2", [False, True])
async def test_missing_provider_configuration_uses_local_image(
    monkeypatch: pytest.MonkeyPatch,
    graph: SimpleNamespace,
    providers: tuple[AsyncMock, Mock, AsyncMock],
    use_v2: bool,
) -> None:
    ideogram, replicate, counter = providers
    _configure_providers(monkeypatch, use_v2, api_key="")
    local_image = io.BytesIO(b"local-image")
    generate_local = Mock(return_value=local_image)
    monkeypatch.setattr(image_gen, "generate_local_agent_image", generate_local)

    result = await image_gen.generate_agent_image(graph, "user-1")  # type: ignore[arg-type]

    assert result is local_image
    generate_local.assert_called_once_with(graph)
    ideogram.assert_not_awaited()
    replicate.assert_not_called()
    # Drawing locally spends no provider key, so it leaves the daily cap alone.
    counter.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_v2", [False, True])
async def test_configured_provider_is_used_without_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
    graph: SimpleNamespace,
    providers: tuple[AsyncMock, Mock, AsyncMock],
    use_v2: bool,
) -> None:
    ideogram, replicate, counter = providers
    _configure_providers(monkeypatch, use_v2, api_key="configured")
    generate_local = Mock()
    monkeypatch.setattr(image_gen, "generate_local_agent_image", generate_local)

    result = await image_gen.generate_agent_image(graph, "user-1")  # type: ignore[arg-type]

    assert result.getvalue() == _provider_jpeg()
    generate_local.assert_not_called()
    counter.assert_awaited_once()
    if use_v2:
        ideogram.assert_awaited_once()
        replicate.assert_not_called()
    else:
        replicate.assert_called_once()
        ideogram.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_v2", [False, True])
async def test_configured_provider_failure_does_not_use_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
    graph: SimpleNamespace,
    providers: tuple[AsyncMock, Mock, AsyncMock],
    use_v2: bool,
) -> None:
    ideogram, replicate, _ = providers
    _configure_providers(monkeypatch, use_v2, api_key="configured")
    generate_local = Mock()
    monkeypatch.setattr(image_gen, "generate_local_agent_image", generate_local)
    ideogram.side_effect = RuntimeError("provider unavailable")
    replicate.side_effect = RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await image_gen.generate_agent_image(graph, "user-1")  # type: ignore[arg-type]

    generate_local.assert_not_called()


def test_local_image_is_deterministic_valid_jpeg(graph: SimpleNamespace) -> None:
    first = image_gen.generate_local_agent_image(graph)  # type: ignore[arg-type]
    second = image_gen.generate_local_agent_image(graph)  # type: ignore[arg-type]

    assert first.getvalue() == second.getvalue()
    assert first.getvalue().startswith(b"\xff\xd8\xff")
    with Image.open(first) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (1024, 768)
        image.verify()


def test_local_image_safely_handles_untrusted_graph_text() -> None:
    graph = SimpleNamespace(
        name="\x00\n<script>alert('x')</script> 🚀 " * 200,
        description="\udcff\r\n../secrets\x07" * 500,
    )

    result = image_gen.generate_local_agent_image(graph)  # type: ignore[arg-type]

    with Image.open(result) as image:
        assert image.format == "JPEG"
        assert image.size == (1024, 768)
        image.verify()


def test_local_image_varies_with_graph_metadata(graph: SimpleNamespace) -> None:
    other_graph = SimpleNamespace(
        name=graph.name,
        description="A materially different automation.",
    )

    first = image_gen.generate_local_agent_image(graph)  # type: ignore[arg-type]
    second = image_gen.generate_local_agent_image(other_graph)  # type: ignore[arg-type]

    assert first.getvalue() != second.getvalue()
