import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from pydantic import SecretStr

from backend.api.features.store import image_gen


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


@pytest.fixture
def graph() -> SimpleNamespace:
    return SimpleNamespace(
        name="Customer Follow-up",
        description="Drafts a helpful response and routes it for review.",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_v2", [False, True])
async def test_missing_provider_configuration_uses_local_image(
    monkeypatch: pytest.MonkeyPatch,
    graph: SimpleNamespace,
    use_v2: bool,
) -> None:
    monkeypatch.setattr(
        image_gen.settings.config, "use_agent_image_generation_v2", use_v2
    )
    monkeypatch.setattr(image_gen.settings.secrets, "replicate_api_key", "")
    monkeypatch.setattr(image_gen.ideogram_credentials, "api_key", SecretStr(""))
    local_image = io.BytesIO(b"local-image")
    generate_local = Mock(return_value=local_image)
    generate_v1 = AsyncMock()
    generate_v2 = AsyncMock()
    monkeypatch.setattr(image_gen, "generate_local_agent_image", generate_local)
    monkeypatch.setattr(image_gen, "generate_agent_image_v1", generate_v1)
    monkeypatch.setattr(image_gen, "generate_agent_image_v2", generate_v2)

    result = await image_gen.generate_agent_image(graph)  # type: ignore[arg-type]

    assert result is local_image
    generate_local.assert_called_once_with(graph)
    generate_v1.assert_not_awaited()
    generate_v2.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_v2", [False, True])
async def test_configured_provider_is_used_without_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
    graph: SimpleNamespace,
    use_v2: bool,
) -> None:
    monkeypatch.setattr(
        image_gen.settings.config, "use_agent_image_generation_v2", use_v2
    )
    monkeypatch.setattr(image_gen.settings.secrets, "replicate_api_key", "configured")
    monkeypatch.setattr(
        image_gen.ideogram_credentials, "api_key", SecretStr("configured")
    )
    provider_image = io.BytesIO(b"provider-image")
    generate_local = Mock()
    generate_v1 = AsyncMock(return_value=provider_image)
    generate_v2 = AsyncMock(return_value=provider_image)
    monkeypatch.setattr(image_gen, "generate_local_agent_image", generate_local)
    monkeypatch.setattr(image_gen, "generate_agent_image_v1", generate_v1)
    monkeypatch.setattr(image_gen, "generate_agent_image_v2", generate_v2)

    result = await image_gen.generate_agent_image(graph)  # type: ignore[arg-type]

    assert result is provider_image
    generate_local.assert_not_called()
    if use_v2:
        generate_v2.assert_awaited_once_with(graph=graph)
        generate_v1.assert_not_awaited()
    else:
        generate_v1.assert_awaited_once_with(agent=graph)
        generate_v2.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_v2", [False, True])
async def test_configured_provider_failure_does_not_use_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
    graph: SimpleNamespace,
    use_v2: bool,
) -> None:
    monkeypatch.setattr(
        image_gen.settings.config, "use_agent_image_generation_v2", use_v2
    )
    monkeypatch.setattr(image_gen.settings.secrets, "replicate_api_key", "configured")
    monkeypatch.setattr(
        image_gen.ideogram_credentials, "api_key", SecretStr("configured")
    )
    generate_local = Mock()
    provider_error = RuntimeError("provider unavailable")
    monkeypatch.setattr(image_gen, "generate_local_agent_image", generate_local)
    monkeypatch.setattr(
        image_gen,
        "generate_agent_image_v1",
        AsyncMock(side_effect=provider_error),
    )
    monkeypatch.setattr(
        image_gen,
        "generate_agent_image_v2",
        AsyncMock(side_effect=provider_error),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await image_gen.generate_agent_image(graph)  # type: ignore[arg-type]

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
