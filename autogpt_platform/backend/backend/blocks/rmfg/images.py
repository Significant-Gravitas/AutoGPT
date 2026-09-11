"""Block that fetches RMFG's rendered pictures of a design or part.

The API's image links need the account's credentials, so a bare URL is of no
use to a downstream block or a person; this block downloads the picture and
hands it on as a platform media file.
"""

import base64

from backend.data.execution import ExecutionContext
from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    MediaFileType,
    SchemaField,
    store_media_file,
)

from ._api import RMFGClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._inputs import credentials_field
from ._testdata import TEST_DESIGN, TEST_PART, TEST_PNG_DATA_URI
from ._types import ImageView, RMFGCredentials

CATEGORIES = {BlockCategory.HARDWARE, BlockCategory.MULTIMEDIA}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RMFGGetImageBlock(Block):
    """Render a design, one of its parts, or a DFM report's configured part."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        design_id: str = SchemaField(description="Design ID from Analyze Design")
        part_id: str = SchemaField(
            description=(
                "Draw only this part, with every hole and bend labelled by ID. "
                "Empty draws the whole design."
            ),
            default="",
        )
        dfm_id: str = SchemaField(
            description=(
                "With part_id, draw the part as configured in this DFM report, "
                "taps, studs, nuts and countersinks marked."
            ),
            default="",
            advanced=True,
        )
        view: ImageView = SchemaField(
            description="Camera angle; flat shows a sheet part's flat pattern.",
            default=ImageView.ISO,
            advanced=False,
        )
        width: int = SchemaField(
            description="Image width in pixels; 0 uses RMFG's default.",
            default=0,
            ge=0,
            le=4096,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        image: MediaFileType = SchemaField(description="The rendered PNG")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="1c1f8d0e-3e57-4c4a-9b7a-2f6a0f4d9e21",
            description="Downloads RMFG's rendered picture of a design or part, with holes and bends labelled",
            categories=CATEGORIES,
            input_schema=RMFGGetImageBlock.Input,
            output_schema=RMFGGetImageBlock.Output,
            test_input={
                "design_id": TEST_DESIGN.id,
                "part_id": TEST_PART.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("image", lambda value: value.startswith(("workspace://", "data:")))
            ],
            test_mock={
                "fetch_image": lambda *args, **kwargs: TEST_PNG_DATA_URI,
                "save_image": lambda *args, **kwargs: TEST_PNG_DATA_URI,
            },
        )

    @staticmethod
    async def fetch_image(
        credentials: RMFGCredentials, input_data: Input
    ) -> MediaFileType:
        """Download the picture and return it as a PNG data URI."""
        content, content_type = await RMFGClient(credentials).get_image(
            input_data.design_id,
            input_data.part_id,
            input_data.dfm_id,
            input_data.view,
            input_data.width,
        )
        # The block always asks for PNG. Anything else, such as the SVG the
        # API can also render or an HTML error page, must not be stored as a
        # media file, so the bytes are checked rather than the header trusted.
        if not content.startswith(PNG_SIGNATURE):
            raise ValueError(
                f"RMFG returned {content_type or 'an unknown type'} instead of a PNG"
            )
        return MediaFileType(
            f"data:image/png;base64,{base64.b64encode(content).decode('ascii')}"
        )

    @staticmethod
    async def save_image(
        image: MediaFileType, execution_context: ExecutionContext
    ) -> MediaFileType:
        return await store_media_file(
            file=image,
            execution_context=execution_context,
            return_format="for_block_output",
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: RMFGCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        image = await self.fetch_image(credentials, input_data)
        yield "image", await self.save_image(image, execution_context)
