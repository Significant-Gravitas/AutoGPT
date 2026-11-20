from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from backend.blocks.ideogram import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    IdeogramModelBlock,
    UpscaleOption,
)
from backend.util.request import Requests

IDEOGRAM_API = "https://api.ideogram.ai"
IMAGE_BYTES = bytes(range(256)) * 1200


class _IdeogramStub:
    def __init__(self) -> None:
        self.base = ""
        self.downloads = 0
        self.upscales: list[dict[str, Any]] = []

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/ideogram-v3/generate", self.generate)
        app.router.add_get("/generated.png", self.download)
        app.router.add_post("/upscale", self.upscale)
        return app

    async def generate(self, request: web.Request) -> web.Response:
        return web.json_response({"data": [{"url": f"{self.base}/generated.png"}]})

    async def download(self, request: web.Request) -> web.Response:
        self.downloads += 1
        return web.Response(body=IMAGE_BYTES, content_type="image/png")

    async def upscale(self, request: web.Request) -> web.Response:
        form = await request.post()
        image = form["image_file"]
        assert isinstance(image, web.FileField)
        self.upscales.append(
            {
                "filename": image.filename,
                "content_type": image.content_type,
                "body": image.file.read(),
                "image_request": form.get("image_request"),
            }
        )
        return web.json_response({"data": [{"url": f"{self.base}/upscaled.png"}]})


def _local_requests(base: str) -> type[Requests]:
    class LocalRequests(Requests):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["trusted_origins"] = [base]
            super().__init__(*args, **kwargs)

        async def request(self, method: str, url: str, *args: Any, **kwargs: Any):
            local_url = url.replace(IDEOGRAM_API, base)
            return await super().request(method, local_url, *args, **kwargs)

    return LocalRequests


@pytest.mark.asyncio
async def test_upscale_sends_the_whole_image_and_returns_the_url(monkeypatch):
    stub = _IdeogramStub()
    async with TestServer(stub.app(), host="127.0.0.1") as server:
        stub.base = f"http://127.0.0.1:{server.port}"
        monkeypatch.setattr(
            "backend.blocks.ideogram.Requests", _local_requests(stub.base)
        )
        block = IdeogramModelBlock()
        outputs = [
            output
            async for output in block.run(
                IdeogramModelBlock.Input.model_validate(
                    {
                        "credentials": TEST_CREDENTIALS_INPUT,
                        "prompt": "a lighthouse at dusk",
                        "upscale": UpscaleOption.AI_UPSCALE,
                    }
                ),
                credentials=TEST_CREDENTIALS,
            )
        ]

    assert outputs == [("result", f"{stub.base}/upscaled.png")]
    assert stub.downloads == 1
    assert len(stub.upscales) == 1
    upload = stub.upscales[0]
    assert upload["filename"] == "image.png"
    assert upload["content_type"] == "image/png"
    assert upload["body"] == IMAGE_BYTES
    assert upload["image_request"] == "{}"
