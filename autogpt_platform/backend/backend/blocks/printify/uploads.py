from pathlib import PurePosixPath
from urllib.parse import urlsplit

from backend.blocks.printify._api import PrintifyClient, PrintifyImage
from backend.blocks.printify._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    PrintifyCredentials,
    PrintifyCredentialsInput,
    printify,
)
from backend.data.execution import ExecutionContext
from backend.sdk import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    MediaFileType,
    SchemaField,
    store_media_file,
)

DEFAULT_FILE_NAME = "design.png"

_TEST_IMAGE = PrintifyImage(
    id="img-cat-sticker-1",
    file_name="cat-sticker.png",
    preview_url="https://images.printify.com/mockup/img-cat-sticker-1.png",
    width=1200,
    height=1200,
)


class PrintifyUploadImageBlock(Block):
    """Add a design to the Printify media library so products can print it."""

    class Input(BlockSchemaInput):
        credentials: PrintifyCredentialsInput = printify.credentials_field(
            description="Your Printify personal access token"
        )
        image: MediaFileType = SchemaField(
            description="The design: a public image URL, an uploaded file or a data URI"
        )
        file_name: str = SchemaField(
            description="Name to file it under in Printify (defaults to the image's name)",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        image_id: str = SchemaField(description="Printify image id, for products")
        preview_url: str = SchemaField(description="Printify's preview of the image")
        width: int = SchemaField(description="Image width in pixels")
        height: int = SchemaField(description="Image height in pixels")

    def __init__(self):
        super().__init__(
            id="845d02d9-cffa-4f37-9aec-7787368ed30d",
            description=(
                "Upload a design image to your Printify media library and get the "
                "image id that Create Printify Product prints."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.MULTIMEDIA},
            input_schema=PrintifyUploadImageBlock.Input,
            output_schema=PrintifyUploadImageBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "image": "https://example.com/designs/cat-sticker.png",
            },
            test_output=[
                ("image_id", _TEST_IMAGE.id),
                ("preview_url", _TEST_IMAGE.preview_url),
                ("width", 1200),
                ("height", 1200),
            ],
            test_mock={
                "upload": lambda credentials, file_name, url, contents: _TEST_IMAGE
            },
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    @staticmethod
    async def upload(
        credentials: PrintifyCredentials,
        file_name: str,
        url: str | None,
        contents: str | None,
    ) -> PrintifyImage:
        return await PrintifyClient(credentials).upload_image(
            file_name, url=url, contents=contents
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: PrintifyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        image = str(input_data.image)
        file_name = input_data.file_name.strip() or _file_name(image)
        if image.startswith(("https://", "http://")):
            # Printify fetches a public URL itself, which is what it recommends
            # for anything over 5 MB.
            uploaded = await self.upload(credentials, file_name, image, None)
        else:
            data_uri = await store_media_file(
                file=input_data.image,
                execution_context=execution_context,
                return_format="for_external_api",
            )
            contents = str(data_uri).split(",", 1)[-1]
            uploaded = await self.upload(credentials, file_name, None, contents)
        yield "image_id", uploaded.id
        yield "preview_url", uploaded.preview_url
        yield "width", uploaded.width
        yield "height", uploaded.height


def _file_name(image: str) -> str:
    if not image.startswith(("https://", "http://")):
        return DEFAULT_FILE_NAME
    name = PurePosixPath(urlsplit(image).path).name
    return name or DEFAULT_FILE_NAME
