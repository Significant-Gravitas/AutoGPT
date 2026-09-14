import io
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from PIL import Image, UnidentifiedImageError

from backend.api.features.library import db as library_db
from backend.api.features.store import image_gen, local_media, media, routes
from backend.blocks.ideogram import TEST_CREDENTIALS, IdeogramModelBlock
from backend.data.graph import GraphBaseMeta
from backend.util.settings import Settings


@pytest.fixture
def generated_media_io(monkeypatch, tmp_path):
    settings = Settings()
    settings.config.use_agent_image_generation_v2 = True
    settings.config.workspace_storage_dir = str(tmp_path / "workspaces")
    settings.config.platform_base_url = ""
    monkeypatch.setattr(image_gen, "settings", settings)
    monkeypatch.setattr(image_gen, "ideogram_credentials", TEST_CREDENTIALS)
    monkeypatch.setattr(media, "Settings", lambda: settings)
    monkeypatch.setattr(local_media, "Settings", lambda: settings)

    objects = {}

    async def upload(bucket, path, content, *, content_type):
        objects[path] = (content, content_type)

    async def metadata(bucket, path):
        if path not in objects:
            raise FileNotFoundError(path)
        return {"contentType": objects[path][1]}

    storage = AsyncMock()
    storage.__aenter__.return_value = storage
    storage.upload.side_effect = upload
    storage.download_metadata.side_effect = metadata
    monkeypatch.setattr(media.async_storage, "Storage", Mock(return_value=storage))
    scanner = AsyncMock()
    monkeypatch.setattr(media, "scan_content_safe", scanner)
    return settings, objects, scanner


@pytest.fixture
def graph_and_library_update(monkeypatch):
    graph = GraphBaseMeta(id="graph-1", name="Agent", description="An agent")
    monkeypatch.setattr(
        routes.backend.data.graph, "get_graph", AsyncMock(return_value=graph)
    )
    update = AsyncMock(return_value=Mock())
    monkeypatch.setattr(
        library_db.prisma.models.LibraryAgent,
        "prisma",
        Mock(return_value=Mock(update=update)),
    )
    return graph, update


@pytest.fixture
def ideogram_response(monkeypatch, provider_format, mode):
    source = io.BytesIO()
    color = (255, 0, 0, 0) if mode == "RGBA" else "red"
    Image.new(mode, (16, 12), color).save(source, format=provider_format)
    source_bytes = source.getvalue()
    generate = AsyncMock(return_value="https://ideogram.example/generated.png")
    monkeypatch.setattr(IdeogramModelBlock, "run_once", generate)
    download = AsyncMock(return_value=httpx.Response(200, content=source_bytes))
    monkeypatch.setattr(image_gen, "Requests", Mock(return_value=Mock(get=download)))
    return source_bytes, generate, download


@pytest.mark.parametrize("caller", ["marketplace", "library"])
@pytest.mark.parametrize("bucket", ["", "test-bucket"])
@pytest.mark.parametrize(
    "provider_format,mode", [("PNG", "RGB"), ("PNG", "RGBA"), ("JPEG", "RGB")]
)
async def test_generated_thumbnail_upload_and_reuse(
    generated_media_io,
    graph_and_library_update,
    ideogram_response,
    caller,
    bucket,
    provider_format,
    mode,
):
    settings, objects, scanner = generated_media_io
    settings.config.media_gcs_bucket_name = bucket
    graph, update = graph_and_library_update
    source_bytes, generate, download = ideogram_response
    expected_path = "users/test-user/images/agent_graph-1.jpeg"
    expected_url = (
        f"https://storage.googleapis.com/{bucket}/{expected_path}"
        if bucket
        else "/api/store/media/test-user/images/agent_graph-1.jpeg"
    )
    for _ in range(2):
        if caller == "marketplace":
            result = await routes.generate_image(graph.id, user_id="test-user")
            assert result.image_url == expected_url
        else:
            result = await library_db.add_generated_agent_image(
                graph, "test-user", "library-agent-1"
            )
            assert result is not None
            update.assert_awaited_with(
                where={"id": "library-agent-1"}, data={"imageUrl": expected_url}
            )

    generate.assert_awaited_once()
    download.assert_awaited_once()
    if bucket:
        uploaded, content_type = objects[expected_path]
        assert content_type == "image/jpeg"
    else:
        uploaded = local_media.get_media_path(
            "test-user", "images", "agent_graph-1.jpeg"
        ).read_bytes()
    scanner.assert_awaited_once_with(uploaded, filename="agent_graph-1.jpeg")
    with Image.open(io.BytesIO(uploaded)) as thumbnail:
        assert thumbnail.format == "JPEG"
        assert thumbnail.size == (16, 12)
        if mode == "RGBA":
            assert thumbnail.getpixel((0, 0)) == (255, 255, 255)
    if provider_format == "JPEG":
        assert uploaded == source_bytes


@pytest.mark.parametrize("payload", [b"not an image", b"\x89PNG\r\n\x1a\n"])
async def test_invalid_generated_image_is_not_uploaded(
    generated_media_io, graph_and_library_update, monkeypatch, payload
):
    _, objects, scanner = generated_media_io
    graph, _ = graph_and_library_update
    monkeypatch.setattr(
        IdeogramModelBlock,
        "run_once",
        AsyncMock(return_value="https://ideogram.example/image.png"),
    )
    download = AsyncMock(return_value=httpx.Response(200, content=payload))
    monkeypatch.setattr(image_gen, "Requests", Mock(return_value=Mock(get=download)))

    with pytest.raises(UnidentifiedImageError):
        await routes.generate_image(graph.id, user_id="test-user")
    scanner.assert_not_awaited()
    assert not objects
    assert not local_media.media_root().exists()


async def test_flux_still_requests_and_preserves_jpeg(
    generated_media_io, graph_and_library_update, monkeypatch
):
    settings, _, _ = generated_media_io
    settings.config.use_agent_image_generation_v2 = False
    monkeypatch.setattr(image_gen.settings.secrets, "replicate_api_key", "test-key")
    graph, _ = graph_and_library_update
    source = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(source, format="JPEG")
    run = Mock(return_value="https://replicate.example/image.jpg")
    monkeypatch.setattr(image_gen, "ReplicateClient", Mock(return_value=Mock(run=run)))
    download = AsyncMock(return_value=httpx.Response(200, content=source.getvalue()))
    monkeypatch.setattr(image_gen, "Requests", Mock(return_value=Mock(get=download)))

    result = await image_gen.generate_agent_image(graph)
    assert result.read() == source.getvalue()
    assert run.call_args.kwargs["input"]["output_format"] == "jpg"
