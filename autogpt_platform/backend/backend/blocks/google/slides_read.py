import asyncio
from enum import Enum

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError
from backend.util.file import sanitize_filename
from backend.util.request import HTTPClientError, HTTPServerError, Requests
from backend.util.type import MediaFileType

from ._auth import GOOGLE_OAUTH_IS_CONFIGURED, TEST_CREDENTIALS, GoogleCredentials
from ._drive import GoogleDriveFile
from ._slides_api import (
    SlideElement,
    SlideSummary,
    build_slides_service,
    flatten_elements,
    get_page,
    parse_slide,
    presentation_field,
    presentation_file,
    presentation_text,
    require_presentation,
    require_slide_id,
    save_to_file_store,
    slide_text,
    slides_error,
    speaker_notes,
)
from ._slides_testdata import (
    TEST_PICKED_PRESENTATION,
    TEST_PRESENTATION,
    TEST_PRESENTATION_FILE,
    TEST_PRESENTATION_ID,
    TEST_REVENUE_SLIDE,
)

PRESENTATION_FIELDS = (
    "presentationId,title,slides(objectId,pageElements,slideProperties(notesPage))"
)


class ThumbnailSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    X_LARGE = "x_large"


THUMBNAIL_SIZES = {
    ThumbnailSize.SMALL: "SMALL",
    ThumbnailSize.MEDIUM: "MEDIUM",
    ThumbnailSize.LARGE: "LARGE",
    ThumbnailSize.X_LARGE: "WIDTH2000_PX",
}


class GoogleSlidesReadPresentationBlock(Block):
    """Read a presentation's slides: titles, text and speaker notes."""

    class Input(BlockSchemaInput):
        presentation: GoogleDriveFile = presentation_field(
            "The Google Slides presentation to read"
        )

    class Output(BlockSchemaOutput):
        title: str = SchemaField(description="The presentation's title")
        text: str = SchemaField(
            description="All slide text and speaker notes as one Markdown document, for AI blocks"
        )
        slides: list[SlideSummary] = SchemaField(
            description="Every slide with its ID, position, title, text and speaker notes"
        )
        slide: SlideSummary = SchemaField(description="Each slide")
        presentation: GoogleDriveFile = SchemaField(
            description="The presentation, for chaining into other Slides blocks"
        )

    def __init__(self):
        slides = [
            SlideSummary(
                slide_id="p",
                index=0,
                title="Q3 Business Review",
                text="Q3 Business Review\nFinance team",
                speaker_notes="Welcome everyone.",
            ),
            SlideSummary(
                slide_id="g2f3c4d5e6_0_0",
                index=1,
                title="Revenue",
                text="Revenue\nRegion | Revenue\nEMEA | $1.2M",
                speaker_notes="Call out the EMEA launch.",
            ),
        ]
        super().__init__(
            id="36733941-1c12-4746-8935-d7ece4ae925b",
            description=(
                "Read a Google Slides presentation: its title and, for each slide, "
                "the slide ID, position, title, text from shapes and tables, and "
                "speaker notes. Also returns the whole deck as one Markdown text."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleSlidesReadPresentationBlock.Input,
            output_schema=GoogleSlidesReadPresentationBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"presentation": TEST_PICKED_PRESENTATION},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("title", "Q3 Business Review"),
                (
                    "text",
                    "# Q3 Business Review\n\n"
                    "## Slide 1 (id: p)\n\nQ3 Business Review\nFinance team\n\n"
                    "Speaker notes:\nWelcome everyone.\n\n"
                    "## Slide 2 (id: g2f3c4d5e6_0_0)\n\n"
                    "Revenue\nRegion | Revenue\nEMEA | $1.2M\n\n"
                    "Speaker notes:\nCall out the EMEA launch.",
                ),
                ("slides", slides),
                ("slide", slides[0]),
                ("slide", slides[1]),
                ("presentation", TEST_PRESENTATION_FILE),
            ],
            test_mock={"_get_presentation": lambda *args, **kwargs: TEST_PRESENTATION},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_presentation(input_data.presentation, self.name, self.id)
        service = build_slides_service(credentials)
        try:
            data = await asyncio.to_thread(self._get_presentation, service, file.id)
        except HttpError as e:
            raise slides_error(e, self.name, self.id) from e
        title = data.get("title", "")
        slides = [
            parse_slide(slide, index)
            for index, slide in enumerate(data.get("slides", []))
        ]
        yield "title", title
        yield "text", presentation_text(title, slides)
        yield "slides", slides
        for slide in slides:
            yield "slide", slide
        yield "presentation", presentation_file(
            file.id, title or file.name, credentials.id
        )

    @staticmethod
    def _get_presentation(service, presentation_id: str) -> dict:
        return (
            service.presentations()
            .get(presentationId=presentation_id, fields=PRESENTATION_FIELDS)
            .execute()
        )


class GoogleSlidesGetSlideBlock(Block):
    """List everything on one slide, with element IDs for batch updates."""

    class Input(BlockSchemaInput):
        presentation: GoogleDriveFile = presentation_field(
            "The Google Slides presentation the slide is in"
        )
        slide_id: str = SchemaField(
            description="The slide's ID (from Google Slides Read Presentation) or a link to the slide"
        )

    class Output(BlockSchemaOutput):
        elements: list[SlideElement] = SchemaField(
            description=(
                "Everything on the slide with its element ID, type and text. "
                "A group is followed by the elements inside it."
            )
        )
        element: SlideElement = SchemaField(description="Each element")
        text: str = SchemaField(description="All text on the slide")
        speaker_notes: str = SchemaField(description="The slide's speaker notes")
        presentation: GoogleDriveFile = SchemaField(
            description="The presentation, for chaining into other Slides blocks"
        )

    def __init__(self):
        elements = [
            SlideElement(
                element_id="g2f3c4d5e6_0_1",
                type="shape",
                shape_type="TEXT_BOX",
                placeholder="TITLE",
                text="Revenue",
            ),
            SlideElement(
                element_id="g2f3c4d5e6_0_2",
                type="table",
                text="Region | Revenue\nEMEA | $1.2M",
            ),
        ]
        super().__init__(
            id="5ff670af-33bb-4e62-b079-396e5ad22bc9",
            description=(
                "Get one slide of a Google Slides presentation by slide ID: every "
                "element on it (text boxes, shapes, tables, images) with its "
                "element ID, type and text, plus the slide's speaker notes."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleSlidesGetSlideBlock.Input,
            output_schema=GoogleSlidesGetSlideBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "presentation": TEST_PICKED_PRESENTATION,
                "slide_id": f"https://docs.google.com/presentation/d/{TEST_PRESENTATION_ID}/edit#slide=id.g2f3c4d5e6_0_0",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("elements", elements),
                ("element", elements[0]),
                ("element", elements[1]),
                ("text", "Revenue\nRegion | Revenue\nEMEA | $1.2M"),
                ("speaker_notes", "Call out the EMEA launch."),
                ("presentation", TEST_PRESENTATION_FILE),
            ],
            test_mock={"_get_page": lambda *args, **kwargs: TEST_REVENUE_SLIDE},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_presentation(input_data.presentation, self.name, self.id)
        slide_id = require_slide_id(input_data.slide_id, self.name, self.id)
        service = build_slides_service(credentials)
        try:
            page = await asyncio.to_thread(self._get_page, service, file.id, slide_id)
        except HttpError as e:
            raise slides_error(e, self.name, self.id) from e
        elements = flatten_elements(page.get("pageElements", []))
        yield "elements", elements
        for element in elements:
            yield "element", element
        yield "text", slide_text(elements)
        yield "speaker_notes", speaker_notes(page)
        yield "presentation", presentation_file(file.id, file.name, credentials.id)

    @staticmethod
    def _get_page(service, presentation_id: str, slide_id: str) -> dict:
        return get_page(service, presentation_id, slide_id)


class GoogleSlidesGetSlideThumbnailBlock(Block):
    """Render one slide as a PNG image."""

    class Input(BlockSchemaInput):
        presentation: GoogleDriveFile = presentation_field(
            "The Google Slides presentation the slide is in"
        )
        slide_id: str = SchemaField(
            description="The slide's ID (from Google Slides Read Presentation) or a link to the slide"
        )
        size: ThumbnailSize = SchemaField(
            description="Image width: small (200 px), medium (800 px), large (1600 px) or x_large (2000 px)",
            default=ThumbnailSize.MEDIUM,
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        thumbnail: MediaFileType = SchemaField(
            description="The slide as a PNG image (a workspace file in CoPilot, a data URI in agents)"
        )
        width: int = SchemaField(description="Image width in pixels")
        height: int = SchemaField(description="Image height in pixels")
        content_url: str = SchemaField(
            description=(
                "Google's link to the image. It expires after about 30 minutes, "
                "and anyone who has it can see the slide."
            )
        )
        presentation: GoogleDriveFile = SchemaField(
            description="The presentation, for chaining into other Slides blocks"
        )

    def __init__(self):
        thumbnail = {
            "contentUrl": "https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdEXAMPLE=s800",
            "width": 800,
            "height": 450,
        }
        super().__init__(
            id="385f22dc-d6ec-4d84-a064-1b8d3f7301ba",
            description=(
                "Render one slide of a Google Slides presentation as a PNG image, "
                "for example to check how a slide looks after editing it."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.MULTIMEDIA},
            input_schema=GoogleSlidesGetSlideThumbnailBlock.Input,
            output_schema=GoogleSlidesGetSlideThumbnailBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "presentation": TEST_PICKED_PRESENTATION,
                "slide_id": "g2f3c4d5e6_0_0",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("thumbnail", "data:image/png;base64,iVBORw0KGgo="),
                ("width", 800),
                ("height", 450),
                ("content_url", thumbnail["contentUrl"]),
                ("presentation", TEST_PRESENTATION_FILE),
            ],
            test_mock={
                "_get_thumbnail": lambda *args, **kwargs: thumbnail,
                "_download": lambda *args, **kwargs: b"\x89PNG\r\n\x1a\n",
                "_store": lambda *args, **kwargs: "data:image/png;base64,iVBORw0KGgo=",
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        file = require_presentation(input_data.presentation, self.name, self.id)
        slide_id = require_slide_id(input_data.slide_id, self.name, self.id)
        service = build_slides_service(credentials)
        try:
            thumbnail = await asyncio.to_thread(
                self._get_thumbnail,
                service,
                file.id,
                slide_id,
                THUMBNAIL_SIZES[input_data.size],
            )
        except HttpError as e:
            raise slides_error(e, self.name, self.id) from e
        try:
            data = await self._download(thumbnail["contentUrl"])
        except (HTTPClientError, HTTPServerError) as e:
            raise BlockExecutionError(
                message=f"Couldn't download the slide image from Google (HTTP {e.status_code}). Try again.",
                block_name=self.name,
                block_id=self.id,
            ) from e
        name = sanitize_filename(f"{file.name or file.id} - slide {slide_id}.png")
        yield "thumbnail", await self._store(data, name, execution_context)
        yield "width", thumbnail.get("width", 0)
        yield "height", thumbnail.get("height", 0)
        yield "content_url", thumbnail["contentUrl"]
        yield "presentation", presentation_file(file.id, file.name, credentials.id)

    @staticmethod
    def _get_thumbnail(service, presentation_id: str, slide_id: str, size: str) -> dict:
        return (
            service.presentations()
            .pages()
            .getThumbnail(
                presentationId=presentation_id,
                pageObjectId=slide_id,
                thumbnailProperties_thumbnailSize=size,
                thumbnailProperties_mimeType="PNG",
            )
            .execute()
        )

    @staticmethod
    async def _download(url: str) -> bytes:
        response = await Requests().get(url)
        return response.content

    @staticmethod
    async def _store(
        data: bytes, name: str, execution_context: ExecutionContext
    ) -> MediaFileType:
        return await save_to_file_store(data, name, execution_context)
