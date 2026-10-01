import asyncio
from enum import Enum
from typing import Any, Optional

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockInputError

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._drive import GoogleDriveFile
from ._slides_api import (
    DRIVE_FILE_SCOPE,
    TITLE_PLACEHOLDERS,
    batch_update,
    build_slides_service,
    get_page,
    presentation_field,
    presentation_file,
    require_presentation,
    slides_error,
)
from ._slides_testdata import (
    TEST_NEW_SLIDE,
    TEST_PICKED_PRESENTATION,
    TEST_PRESENTATION_FILE,
    TEST_PRESENTATION_ID,
)

BODY_PLACEHOLDERS = ("BODY", "SUBTITLE")
_LAYOUT_HINT = (
    "Presentations whose theme came from PowerPoint may not have Google's "
    "built-in layouts."
)


class SlideLayout(str, Enum):
    """Google's built-in slide layouts."""

    TITLE = "title"
    TITLE_AND_BODY = "title_and_body"
    TITLE_AND_TWO_COLUMNS = "title_and_two_columns"
    TITLE_ONLY = "title_only"
    SECTION_HEADER = "section_header"
    SECTION_TITLE_AND_DESCRIPTION = "section_title_and_description"
    ONE_COLUMN_TEXT = "one_column_text"
    MAIN_POINT = "main_point"
    BIG_NUMBER = "big_number"
    CAPTION_ONLY = "caption_only"
    BLANK = "blank"


class GoogleSlidesCreatePresentationBlock(Block):
    """Create an empty Google Slides presentation."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField([DRIVE_FILE_SCOPE])
        title: str = SchemaField(description="Title of the new presentation")

    class Output(BlockSchemaOutput):
        presentation: GoogleDriveFile = SchemaField(
            description="The new presentation, for chaining into other Slides blocks"
        )

    def __init__(self):
        super().__init__(
            id="4a3c647e-e65f-4322-bb95-f996f678f594",
            description=(
                "Create a new Google Slides presentation with the given title, in "
                "the root of My Drive. It starts with one empty title slide."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleSlidesCreatePresentationBlock.Input,
            output_schema=GoogleSlidesCreatePresentationBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "title": "Q3 Business Review",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("presentation", TEST_PRESENTATION_FILE)],
            test_mock={
                "_create": lambda *args, **kwargs: {
                    "presentationId": TEST_PRESENTATION_ID,
                    "title": "Q3 Business Review",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.title.strip():
            raise BlockInputError(
                message="Give the new presentation a title.",
                block_name=self.name,
                block_id=self.id,
            )
        service = build_slides_service(credentials)
        try:
            created = await asyncio.to_thread(self._create, service, input_data.title)
        except HttpError as e:
            raise slides_error(e, self.name, self.id) from e
        yield "presentation", presentation_file(
            created["presentationId"],
            created.get("title", input_data.title),
            credentials.id,
        )

    @staticmethod
    def _create(service, title: str) -> dict:
        return (
            service.presentations()
            .create(body={"title": title}, fields="presentationId,title")
            .execute()
        )


class GoogleSlidesAddSlideBlock(Block):
    """Add a slide with a built-in layout and fill in its title and body."""

    class Input(BlockSchemaInput):
        presentation: GoogleDriveFile = presentation_field(
            "The Google Slides presentation to add a slide to", edit=True
        )
        layout: SlideLayout = SchemaField(
            description="Which of Google's built-in layouts the slide uses",
            default=SlideLayout.TITLE_AND_BODY,
            advanced=False,
        )
        title: str = SchemaField(
            description="The slide's title",
            default="",
            advanced=False,
        )
        body: str = SchemaField(
            description=(
                "The slide's body text, one paragraph or bullet per line. On the "
                "title layout it goes in the subtitle."
            ),
            default="",
            advanced=False,
        )
        index: Optional[int] = SchemaField(
            description=(
                "Where to put the slide, counting from 0 (0 makes it the first "
                "slide). Empty adds it at the end."
            ),
            default=None,
            ge=0,
        )

    class Output(BlockSchemaOutput):
        slide_id: str = SchemaField(description="The new slide's ID")
        presentation: GoogleDriveFile = SchemaField(
            description="The presentation, for chaining into other Slides blocks"
        )

    def __init__(self):
        super().__init__(
            id="6d233a70-8d49-4863-bed2-5c2e7bde760e",
            description=(
                "Add a slide to a Google Slides presentation using one of Google's "
                "built-in layouts, such as title and body, and fill in its title "
                "and body text."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleSlidesAddSlideBlock.Input,
            output_schema=GoogleSlidesAddSlideBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "presentation": TEST_PICKED_PRESENTATION,
                "layout": SlideLayout.TITLE_AND_BODY,
                "title": "Next steps",
                "body": "Hire two engineers\nLaunch in EMEA",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("slide_id", TEST_NEW_SLIDE["objectId"]),
                ("presentation", TEST_PRESENTATION_FILE),
            ],
            test_mock={
                "_create_slide": lambda *args, **kwargs: TEST_NEW_SLIDE["objectId"],
                "_get_page": lambda *args, **kwargs: TEST_NEW_SLIDE,
                "_batch_update": lambda *args, **kwargs: {"replies": [{}, {}]},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_presentation(input_data.presentation, self.name, self.id)
        if input_data.layout == SlideLayout.BLANK and (
            input_data.title or input_data.body
        ):
            raise BlockInputError(
                message="The blank layout has no title or body. Pick another layout, or leave the title and body empty.",
                block_name=self.name,
                block_id=self.id,
            )
        service = build_slides_service(credentials)
        try:
            slide_id = await asyncio.to_thread(
                self._create_slide,
                service,
                file.id,
                input_data.layout,
                input_data.index,
            )
            if input_data.title or input_data.body:
                await self._fill(service, file.id, slide_id, input_data)
        except HttpError as e:
            raise slides_error(e, self.name, self.id, _LAYOUT_HINT) from e
        yield "slide_id", slide_id
        yield "presentation", presentation_file(file.id, file.name, credentials.id)

    async def _fill(
        self, service, presentation_id: str, slide_id: str, input_data: Input
    ) -> None:
        """Put the title and body into the new slide's placeholders. If the
        layout has no place for them, remove the slide again."""
        page = await asyncio.to_thread(
            self._get_page, service, presentation_id, slide_id
        )
        requests, missing = text_requests(page, input_data.title, input_data.body)
        if missing:
            await asyncio.to_thread(
                self._batch_update,
                service,
                presentation_id,
                [{"deleteObject": {"objectId": slide_id}}],
            )
            layout = input_data.layout.value.replace("_", " ")
            raise BlockInputError(
                message=(
                    f"This presentation's {layout} layout has no "
                    f"{' or '.join(missing)} placeholder, so no slide was added. "
                    f"Pick another layout, or leave the {' and '.join(missing)} empty."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        await asyncio.to_thread(self._batch_update, service, presentation_id, requests)

    @staticmethod
    def _create_slide(
        service, presentation_id: str, layout: SlideLayout, index: Optional[int]
    ) -> str:
        create: dict[str, Any] = {
            "slideLayoutReference": {"predefinedLayout": layout.value.upper()}
        }
        if index is not None:
            create["insertionIndex"] = index
        response = batch_update(service, presentation_id, [{"createSlide": create}])
        return response["replies"][0]["createSlide"]["objectId"]

    @staticmethod
    def _get_page(service, presentation_id: str, slide_id: str) -> dict:
        return get_page(service, presentation_id, slide_id)

    @staticmethod
    def _batch_update(service, presentation_id: str, requests: list[dict]) -> dict:
        return batch_update(service, presentation_id, requests)


def text_requests(
    page: dict[str, Any], title: str, body: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build insertText requests for a new slide's title and body, and name any
    part that has no placeholder to go in."""
    requests: list[dict[str, Any]] = []
    missing: list[str] = []
    for part, text, types in (
        ("title", title, TITLE_PLACEHOLDERS),
        ("body", body, BODY_PLACEHOLDERS),
    ):
        if not text:
            continue
        object_id = find_placeholder(page, types)
        if object_id:
            requests.append({"insertText": {"objectId": object_id, "text": text}})
        else:
            missing.append(part)
    return requests, missing


def find_placeholder(page: dict[str, Any], types: tuple[str, ...]) -> Optional[str]:
    """The first placeholder of the first type in `types` that the page has,
    lowest placeholder index first (the left column of a two-column layout)."""
    placeholders = [
        (element["shape"]["placeholder"], element["objectId"])
        for element in page.get("pageElements", [])
        if "placeholder" in element.get("shape", {})
    ]
    for wanted in types:
        matches = sorted(
            (placeholder.get("index", 0), object_id)
            for placeholder, object_id in placeholders
            if placeholder.get("type") == wanted
        )
        if matches:
            return matches[0][1]
    return None
