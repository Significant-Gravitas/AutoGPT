import base64
import binascii

from backend.blocks.etsy._api import EtsyClient, EtsyListingImage
from backend.blocks.etsy._config import (
    ETSY_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    EtsyCredentials,
    EtsyCredentialsInput,
    etsy,
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

_TEST_LISTING_ID = 1500000001
_TEST_IMAGE = EtsyListingImage(
    listing_image_id=4400001,
    url_fullxfull="https://i.etsystatic.com/listing/4400001/il_fullxfull.png",
)


class EtsyUploadListingImageBlock(Block):
    """Add a photo or mockup to an Etsy listing."""

    class Input(BlockSchemaInput):
        credentials: EtsyCredentialsInput = etsy.credentials_field(
            description="Your Etsy shop"
        )
        shop_id: int = SchemaField(description="Shop id from Get My Etsy Shop")
        listing_id: int = SchemaField(description="Listing to add the image to")
        image: MediaFileType = SchemaField(
            description="The photo or mockup: an image URL, an uploaded file or a data URI"
        )
        rank: int = SchemaField(
            description="Position among the listing's photos; 1 is the main photo",
            default=1,
        )
        alt_text: str = SchemaField(
            description="Describes the image for screen readers", default=""
        )

    class Output(BlockSchemaOutput):
        listing_image_id: int = SchemaField(description="The uploaded image's id")
        image_url: str = SchemaField(description="Full-size image on Etsy")

    def __init__(self):
        super().__init__(
            id="65423e00-fcfa-49eb-8f40-b913e034cea6",
            description=(
                "Upload a photo or mockup to an Etsy listing. A listing needs at "
                "least one image before it can be published."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.MULTIMEDIA},
            input_schema=EtsyUploadListingImageBlock.Input,
            output_schema=EtsyUploadListingImageBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "shop_id": 5550123,
                "listing_id": _TEST_LISTING_ID,
                "image": "data:image/png;base64,iVBORw0KGgo=",
            },
            test_output=[
                ("listing_image_id", _TEST_IMAGE.listing_image_id),
                ("image_url", _TEST_IMAGE.url_fullxfull),
            ],
            test_mock={
                "upload": lambda credentials, shop_id, listing_id, image, form: (
                    _TEST_IMAGE
                ),
                "load_image": lambda image, execution_context: (
                    "image.png",
                    b"png",
                    "image/png",
                ),
            },
            effect=BlockEffect.EXTERNAL_JUDGED,
            disabled=not ETSY_OAUTH_IS_CONFIGURED,
        )

    @staticmethod
    async def load_image(
        image: MediaFileType, execution_context: ExecutionContext
    ) -> tuple[str, bytes, str]:
        """The image as (file name, bytes, MIME type), fetched safely."""
        data_uri = await store_media_file(
            file=image,
            execution_context=execution_context,
            return_format="for_external_api",
        )
        return decode_data_uri(str(data_uri))

    @staticmethod
    async def upload(
        credentials: EtsyCredentials,
        shop_id: int,
        listing_id: int,
        image: tuple[str, bytes, str],
        form: dict[str, str],
    ) -> EtsyListingImage:
        return await EtsyClient(credentials).upload_listing_image(
            shop_id, listing_id, image, form
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: EtsyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        image = await self.load_image(input_data.image, execution_context)
        form = {"rank": str(max(1, input_data.rank))}
        if input_data.alt_text:
            form["alt_text"] = input_data.alt_text[:500]
        uploaded = await self.upload(
            credentials, input_data.shop_id, input_data.listing_id, image, form
        )
        yield "listing_image_id", uploaded.listing_image_id
        yield "image_url", uploaded.url_fullxfull


def decode_data_uri(data_uri: str) -> tuple[str, bytes, str]:
    header, _, payload = data_uri.partition(",")
    if not header.startswith("data:") or ";base64" not in header:
        raise ValueError("The image could not be read. Try a PNG or JPEG file.")
    mime_type = header.removeprefix("data:").split(";", 1)[0] or "image/png"
    try:
        content = base64.b64decode(payload, validate=True)
    except binascii.Error as e:
        raise ValueError("The image could not be read. Try a PNG or JPEG file.") from e
    extension = mime_type.rsplit("/", 1)[-1].replace("jpeg", "jpg")
    return f"image.{extension}", content, mime_type
