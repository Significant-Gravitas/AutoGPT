import asyncio
from typing import Any

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

from ._auth import GOOGLE_OAUTH_IS_CONFIGURED, TEST_CREDENTIALS, GoogleCredentials
from ._drive import GoogleDriveFile
from ._slides_api import (
    batch_update,
    build_slides_service,
    get_page,
    presentation_field,
    presentation_file,
    require_presentation,
    require_slide_id,
    slides_error,
    speaker_notes,
    speaker_notes_id,
)
from ._slides_testdata import (
    TEST_PICKED_PRESENTATION,
    TEST_PRESENTATION_FILE,
    TEST_REVENUE_SLIDE,
)


class GoogleSlidesBatchUpdateBlock(Block):
    """Send raw Slides API batchUpdate requests."""

    class Input(BlockSchemaInput):
        presentation: GoogleDriveFile = presentation_field(
            "The Google Slides presentation to change", edit=True
        )
        requests: list[dict[str, Any]] = SchemaField(
            description=(
                "Slides API batchUpdate requests, each an object with one request "
                'type, e.g. {"deleteObject": {"objectId": "g2f3c4d5e6_0_0"}}. See '
                "https://developers.google.com/workspace/slides/api/reference/rest/v1/presentations/request"
            )
        )

    class Output(BlockSchemaOutput):
        replies: list[dict[str, Any]] = SchemaField(
            description=(
                "One reply per request, in order, such as the IDs of created "
                "objects. Requests with nothing to report get an empty object."
            )
        )
        presentation: GoogleDriveFile = SchemaField(
            description="The presentation, for chaining into other Slides blocks"
        )

    def __init__(self):
        requests = [
            {
                "createShape": {
                    "objectId": "summary_box",
                    "shapeType": "TEXT_BOX",
                    "elementProperties": {"pageObjectId": "g2f3c4d5e6_0_0"},
                }
            },
            {"insertText": {"objectId": "summary_box", "text": "Up 12% on Q2"}},
        ]
        replies = [{"createShape": {"objectId": "summary_box"}}, {}]
        super().__init__(
            id="57fd5cc1-8cef-4a63-a67c-7bb062aa8768",
            description=(
                "Change a Google Slides presentation with Slides API batchUpdate "
                "requests: add shapes, tables, images and slides, insert or delete "
                "text, restyle, reorder or delete objects. All requests succeed "
                "together or none are applied."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleSlidesBatchUpdateBlock.Input,
            output_schema=GoogleSlidesBatchUpdateBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"presentation": TEST_PICKED_PRESENTATION, "requests": requests},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("replies", replies),
                ("presentation", TEST_PRESENTATION_FILE),
            ],
            test_mock={"_batch_update": lambda *args, **kwargs: {"replies": replies}},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_presentation(input_data.presentation, self.name, self.id)
        if not input_data.requests:
            raise BlockInputError(
                message="Add at least one request.",
                block_name=self.name,
                block_id=self.id,
            )
        service = build_slides_service(credentials)
        try:
            response = await asyncio.to_thread(
                self._batch_update, service, file.id, input_data.requests
            )
        except HttpError as e:
            raise slides_error(e, self.name, self.id) from e
        yield "replies", response.get("replies", [])
        yield "presentation", presentation_file(file.id, file.name, credentials.id)

    @staticmethod
    def _batch_update(service, presentation_id: str, requests: list[dict]) -> dict:
        return batch_update(service, presentation_id, requests)


class GoogleSlidesReplaceAllTextBlock(Block):
    """Fill in a template: replace several placeholders in one go."""

    class Input(BlockSchemaInput):
        presentation: GoogleDriveFile = presentation_field(
            "The Google Slides presentation to change", edit=True
        )
        replacements: dict[str, str] = SchemaField(
            description=(
                "Text to find, and what to replace it with, e.g. "
                '{"{{client}}": "Acme Corp", "{{date}}": "1 October 2026"}'
            )
        )
        match_case: bool = SchemaField(
            description="Only replace text whose upper and lower case match exactly",
            default=True,
        )
        slide_ids: list[str] = SchemaField(
            description="Only replace text on these slides (IDs or links). Empty means every slide.",
            default_factory=list,
        )

    class Output(BlockSchemaOutput):
        occurrences_changed: int = SchemaField(
            description="How many replacements were made in total"
        )
        occurrences: dict[str, int] = SchemaField(
            description="How many times each text was replaced; 0 means it wasn't found"
        )
        presentation: GoogleDriveFile = SchemaField(
            description="The presentation, for chaining into other Slides blocks"
        )

    def __init__(self):
        super().__init__(
            id="306e334c-c6bc-4ee2-9abe-122556d18069",
            description=(
                "Replace text everywhere in a Google Slides presentation, with "
                "several find-and-replace pairs at once. Use it to fill in a "
                "template's placeholders, such as {{client}}, and see how often "
                "each one was replaced."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleSlidesReplaceAllTextBlock.Input,
            output_schema=GoogleSlidesReplaceAllTextBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "presentation": TEST_PICKED_PRESENTATION,
                "replacements": {"{{client}}": "Acme Corp", "{{quarter}}": "Q3"},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("occurrences_changed", 4),
                ("occurrences", {"{{client}}": 3, "{{quarter}}": 1}),
                ("presentation", TEST_PRESENTATION_FILE),
            ],
            test_mock={
                "_batch_update": lambda *args, **kwargs: {
                    "replies": [
                        {"replaceAllText": {"occurrencesChanged": 3}},
                        {"replaceAllText": {"occurrencesChanged": 1}},
                    ]
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_presentation(input_data.presentation, self.name, self.id)
        if not input_data.replacements or "" in input_data.replacements:
            raise BlockInputError(
                message="Add at least one text to find, and don't leave any of them empty.",
                block_name=self.name,
                block_id=self.id,
            )
        slide_ids = [
            require_slide_id(value, self.name, self.id)
            for value in input_data.slide_ids
        ]
        requests = replace_text_requests(
            input_data.replacements, input_data.match_case, slide_ids
        )
        service = build_slides_service(credentials)
        try:
            response = await asyncio.to_thread(
                self._batch_update, service, file.id, requests
            )
        except HttpError as e:
            raise slides_error(e, self.name, self.id) from e
        counts = {
            find: reply.get("replaceAllText", {}).get("occurrencesChanged", 0)
            for find, reply in zip(input_data.replacements, response.get("replies", []))
        }
        yield "occurrences_changed", sum(counts.values())
        yield "occurrences", counts
        yield "presentation", presentation_file(file.id, file.name, credentials.id)

    @staticmethod
    def _batch_update(service, presentation_id: str, requests: list[dict]) -> dict:
        return batch_update(service, presentation_id, requests)


class GoogleSlidesSetSpeakerNotesBlock(Block):
    """Set or replace one slide's speaker notes."""

    class Input(BlockSchemaInput):
        presentation: GoogleDriveFile = presentation_field(
            "The Google Slides presentation the slide is in", edit=True
        )
        slide_id: str = SchemaField(
            description="The slide's ID (from Google Slides Read Presentation) or a link to the slide"
        )
        notes: str = SchemaField(
            description="The new speaker notes. They replace the slide's current notes; empty text clears them."
        )

    class Output(BlockSchemaOutput):
        presentation: GoogleDriveFile = SchemaField(
            description="The presentation, for chaining into other Slides blocks"
        )

    def __init__(self):
        super().__init__(
            id="04409a5c-771d-4160-b5a6-e07c34519299",
            description=(
                "Set the speaker notes of one slide in a Google Slides "
                "presentation, replacing any notes it already has."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleSlidesSetSpeakerNotesBlock.Input,
            output_schema=GoogleSlidesSetSpeakerNotesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "presentation": TEST_PICKED_PRESENTATION,
                "slide_id": TEST_REVENUE_SLIDE["objectId"],
                "notes": "Revenue grew 12%. Pause for questions.",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("presentation", TEST_PRESENTATION_FILE)],
            test_mock={
                "_get_page": lambda *args, **kwargs: TEST_REVENUE_SLIDE,
                "_batch_update": lambda *args, **kwargs: {"replies": [{}, {}]},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_presentation(input_data.presentation, self.name, self.id)
        slide_id = require_slide_id(input_data.slide_id, self.name, self.id)
        service = build_slides_service(credentials)
        try:
            page = await asyncio.to_thread(self._get_page, service, file.id, slide_id)
            notes_id = speaker_notes_id(page)
            if not notes_id:
                raise BlockInputError(
                    message=f"'{slide_id}' is not a slide, so it has no speaker notes. Use a slide ID from Google Slides Read Presentation.",
                    block_name=self.name,
                    block_id=self.id,
                )
            requests = notes_requests(notes_id, speaker_notes(page), input_data.notes)
            if requests:
                await asyncio.to_thread(self._batch_update, service, file.id, requests)
        except HttpError as e:
            raise slides_error(e, self.name, self.id) from e
        yield "presentation", presentation_file(file.id, file.name, credentials.id)

    @staticmethod
    def _get_page(service, presentation_id: str, slide_id: str) -> dict:
        return get_page(service, presentation_id, slide_id)

    @staticmethod
    def _batch_update(service, presentation_id: str, requests: list[dict]) -> dict:
        return batch_update(service, presentation_id, requests)


def replace_text_requests(
    replacements: dict[str, str], match_case: bool, slide_ids: list[str]
) -> list[dict[str, Any]]:
    only_on = {"pageObjectIds": slide_ids} if slide_ids else {}
    return [
        {
            "replaceAllText": {
                "containsText": {"text": find, "matchCase": match_case},
                "replaceText": replace,
                **only_on,
            }
        }
        for find, replace in replacements.items()
    ]


def notes_requests(notes_id: str, current: str, notes: str) -> list[dict[str, Any]]:
    """Clear the notes shape if it has text, then insert the new notes. Google
    creates the shape on insert when the slide has never had notes."""
    clear = [{"deleteText": {"objectId": notes_id, "textRange": {"type": "ALL"}}}]
    insert = [
        {"insertText": {"objectId": notes_id, "insertionIndex": 0, "text": notes}}
    ]
    return (clear if current else []) + (insert if notes else [])
