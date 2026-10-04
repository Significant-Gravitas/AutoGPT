"""Unit tests for the Google Slides blocks.

The blocks' own test_input/test_mock cases mock the API calls away. These run
the blocks against the real Slides API client, with only the HTTP layer
replaced by canned responses, so they check the requests the blocks send,
how they read the responses, and how they report errors.
"""

import json
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import googleapiclient
import httplib2
import pytest
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import HttpMockSequence

from backend.blocks.google import _slides_api, slides_create, slides_edit, slides_read
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._drive import GoogleDriveFile
from backend.blocks.google._slides_api import (
    DRIVE_FILE_SCOPE,
    PRESENTATIONS_SCOPE,
    SlideSummary,
    flatten_elements,
    parse_slide,
    plain_text,
    presentation_file,
    presentation_text,
    require_presentation,
    require_slide_id,
    slides_error,
)
from backend.blocks.google._slides_testdata import (
    TEST_NEW_SLIDE,
    TEST_PICKED_PRESENTATION,
    TEST_PRESENTATION,
    TEST_REVENUE_SLIDE,
    notes_properties,
    placeholder_shape,
    text_content,
)
from backend.blocks.google.slides_create import (
    GoogleSlidesAddSlideBlock,
    GoogleSlidesCreatePresentationBlock,
    find_placeholder,
)
from backend.blocks.google.slides_edit import (
    GoogleSlidesBatchUpdateBlock,
    GoogleSlidesReplaceAllTextBlock,
    GoogleSlidesSetSpeakerNotesBlock,
)
from backend.blocks.google.slides_read import (
    GoogleSlidesGetSlideBlock,
    GoogleSlidesGetSlideThumbnailBlock,
    GoogleSlidesReadPresentationBlock,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.file import clean_exec_files, get_exec_file_path
from backend.util.json import to_dict
from backend.util.request import HTTPClientError

SLIDE_LINK = (
    "https://docs.google.com/presentation/d/1QnV2p8b4TmZ7xYc3dRk9LsWq0aEfGhJiKlMnOpQrStU"
    "/edit#slide=id.g2f3c4d5e6_0_0"
)


class _RecordingHttp(HttpMockSequence):
    """Replays canned Slides API responses and records every request."""

    def __init__(self, *responses: tuple[int, dict[str, Any]]):
        super().__init__(
            [({"status": str(status)}, json.dumps(body)) for status, body in responses]
        )
        self.calls: list[tuple[str, str, Any]] = []

    def request(
        self,
        uri,
        method="GET",
        body=None,
        headers=None,
        redirections=1,
        connection_type=None,
    ):
        self.calls.append((method, uri, json.loads(body) if body else None))
        return super().request(
            uri, method, body, headers, redirections, connection_type
        )


@pytest.fixture
def slides_api(monkeypatch):
    """Serve canned responses to a block module's Slides client."""

    def install(module, *responses: tuple[int, dict[str, Any]]) -> _RecordingHttp:
        http = _RecordingHttp(*responses)
        service = build("slides", "v1", http=http, cache_discovery=False)
        monkeypatch.setattr(module, "build_slides_service", lambda _: service)
        return http

    return install


async def _run(block, context: ExecutionContext | None = None, **inputs):
    input_data = block.input_schema.model_validate(inputs)
    return [
        output
        async for output in block.run(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=context or ExecutionContext(),
        )
    ]


def _query(uri: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(uri).query)


def _error(status: int, message: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"code": status, "message": message}}


@pytest.mark.asyncio
async def test_read_presentation_asks_only_for_slide_content(slides_api):
    http = slides_api(slides_read, (200, TEST_PRESENTATION))
    outputs = await _run(
        GoogleSlidesReadPresentationBlock(), presentation=TEST_PICKED_PRESENTATION
    )
    method, uri, _ = http.calls[0]
    assert method == "GET"
    assert urlparse(uri).path.endswith(
        f"/presentations/{TEST_PRESENTATION['presentationId']}"
    )
    assert _query(uri)["fields"] == [slides_read.PRESENTATION_FIELDS]
    names = [name for name, _ in outputs]
    assert names == ["title", "text", "slides", "slide", "slide", "presentation"]


def test_flatten_elements_lists_group_contents_after_the_group():
    elements = flatten_elements(
        [
            {
                "objectId": "group1",
                "elementGroup": {
                    "children": [
                        placeholder_shape("inner", "BODY", "Inside the group"),
                        {"objectId": "logo", "image": {"contentUrl": "https://x"}},
                    ]
                },
            },
            {"objectId": "art", "wordArt": {"renderedText": "Big news"}},
            {"objectId": "odd", "somethingNew": {}},
        ]
    )
    assert [(e.element_id, e.type, e.text) for e in elements] == [
        ("group1", "group", ""),
        ("inner", "shape", "Inside the group"),
        ("logo", "image", ""),
        ("art", "word_art", "Big news"),
        ("odd", "unknown", ""),
    ]


def test_plain_text_joins_runs_and_auto_text_and_keeps_soft_line_breaks():
    text = {
        "textElements": [
            {"paragraphMarker": {}},
            {"textRun": {"content": "Slide "}},
            {"autoText": {"type": "SLIDE_NUMBER", "content": "3"}},
            {"textRun": {"content": " of 9\x0bsecond line\n"}},
        ]
    }
    assert plain_text(text) == "Slide 3 of 9\nsecond line"
    assert plain_text(None) == ""


def test_table_text_separates_cells_and_skips_empty_rows():
    table = {
        "objectId": "t1",
        "table": {
            "tableRows": [
                {
                    "tableCells": [
                        {"text": text_content("A")},
                        {"text": text_content("B")},
                    ]
                },
                {"tableCells": [{}, {}]},
            ]
        },
    }
    assert flatten_elements([table])[0].text == "A | B"


def test_parse_slide_without_notes_or_title_text():
    slide = {
        "objectId": "g9",
        "pageElements": [
            placeholder_shape("t", "TITLE"),
            placeholder_shape("b", "BODY", "Body only"),
        ],
        "slideProperties": notes_properties("n9"),
    }
    assert parse_slide(slide, 4) == SlideSummary(
        slide_id="g9", index=4, title="", text="Body only", speaker_notes=""
    )


def test_presentation_text_without_title_or_notes():
    slides = [SlideSummary(slide_id="p", index=0, text="Hello")]
    assert presentation_text("", slides) == "## Slide 1 (id: p)\n\nHello"


@pytest.mark.asyncio
async def test_get_slide_fetches_the_linked_slide(slides_api):
    http = slides_api(slides_read, (200, TEST_REVENUE_SLIDE))
    outputs = dict(
        await _run(
            GoogleSlidesGetSlideBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            slide_id=SLIDE_LINK,
        )
    )
    assert urlparse(http.calls[0][1]).path.endswith("/pages/g2f3c4d5e6_0_0")
    assert outputs["speaker_notes"] == "Call out the EMEA launch."
    assert outputs["text"] == "Revenue\nRegion | Revenue\nEMEA | $1.2M"


class _Response:
    def __init__(self, content: bytes):
        self.content = content


@pytest.mark.asyncio
async def test_thumbnail_is_a_png_at_the_chosen_width_saved_under_a_readable_name(
    slides_api, monkeypatch
):
    content_url = "https://lh7-rt.googleusercontent.com/slidesz/AGV_vUd=s1600"
    http = slides_api(
        slides_read, (200, {"contentUrl": content_url, "width": 1600, "height": 900})
    )
    downloads: list[str] = []

    class _Requests:
        async def get(self, url: str) -> _Response:
            downloads.append(url)
            return _Response(b"\x89PNG\r\n\x1a\n")

    stored: dict[str, Any] = {}

    async def fake_store(file, execution_context, return_format):
        stored.update(file=file, return_format=return_format)
        return "data:image/png;base64,iVBORw0KGgo="

    monkeypatch.setattr(slides_read, "Requests", _Requests)
    monkeypatch.setattr(_slides_api, "store_media_file", fake_store)
    context = ExecutionContext(user_id="user-1", graph_exec_id=str(uuid.uuid4()))
    try:
        outputs = dict(
            await _run(
                GoogleSlidesGetSlideThumbnailBlock(),
                context,
                presentation=TEST_PICKED_PRESENTATION,
                slide_id="g2f3c4d5e6_0_0",
                size="large",
            )
        )
        query = _query(http.calls[0][1])
        assert query["thumbnailProperties.thumbnailSize"] == ["LARGE"]
        assert query["thumbnailProperties.mimeType"] == ["PNG"]
        assert downloads == [content_url]
        assert stored == {
            "file": "Q3 Business Review - slide g2f3c4d5e6_0_0.png",
            "return_format": "for_block_output",
        }
        saved = Path(get_exec_file_path(context.graph_exec_id or "", stored["file"]))
        assert saved.read_bytes() == b"\x89PNG\r\n\x1a\n"
    finally:
        clean_exec_files(context.graph_exec_id or "")
    assert outputs["thumbnail"] == "data:image/png;base64,iVBORw0KGgo="
    assert (outputs["width"], outputs["height"]) == (1600, 900)
    assert outputs["content_url"] == content_url


@pytest.mark.asyncio
async def test_thumbnail_download_failure_is_reported(slides_api, monkeypatch):
    slides_api(
        slides_read,
        (200, {"contentUrl": "https://lh7-rt.googleusercontent.com/x", "width": 1}),
    )

    class _Requests:
        async def get(self, url: str) -> _Response:
            raise HTTPClientError("HTTP 403 Error: Forbidden", 403)

    monkeypatch.setattr(slides_read, "Requests", _Requests)
    with pytest.raises(BlockExecutionError, match="HTTP 403"):
        await _run(
            GoogleSlidesGetSlideThumbnailBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            slide_id="p",
        )


@pytest.mark.asyncio
async def test_create_presentation_sends_only_the_title(slides_api):
    http = slides_api(
        slides_create, (200, {"presentationId": "new123", "title": "Board deck"})
    )
    outputs = await _run(
        GoogleSlidesCreatePresentationBlock(),
        credentials=TEST_CREDENTIALS_INPUT,
        title="Board deck",
    )
    method, uri, body = http.calls[0]
    assert (method, body) == ("POST", {"title": "Board deck"})
    assert _query(uri)["fields"] == ["presentationId,title"]
    assert outputs == [
        ("presentation", presentation_file("new123", "Board deck", TEST_CREDENTIALS.id))
    ]


@pytest.mark.asyncio
async def test_create_presentation_needs_a_title(slides_api):
    http = slides_api(slides_create)
    with pytest.raises(BlockInputError, match="title"):
        await _run(
            GoogleSlidesCreatePresentationBlock(),
            credentials=TEST_CREDENTIALS_INPUT,
            title="  ",
        )
    assert http.calls == []


@pytest.mark.asyncio
async def test_batch_update_sends_requests_as_given(slides_api):
    requests = [{"deleteObject": {"objectId": "g2f3c4d5e6_0_2"}}]
    http = slides_api(slides_edit, (200, {"replies": [{}]}))
    outputs = dict(
        await _run(
            GoogleSlidesBatchUpdateBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            requests=requests,
        )
    )
    method, uri, body = http.calls[0]
    assert method == "POST" and uri.split("?")[0].endswith(":batchUpdate")
    assert body == {"requests": requests}
    assert outputs["replies"] == [{}]


@pytest.mark.asyncio
async def test_batch_update_needs_requests(slides_api):
    http = slides_api(slides_edit)
    with pytest.raises(BlockInputError, match="at least one request"):
        await _run(
            GoogleSlidesBatchUpdateBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            requests=[],
        )
    assert http.calls == []


@pytest.mark.asyncio
async def test_replace_all_text_counts_each_placeholder(slides_api):
    http = slides_api(
        slides_edit,
        (
            200,
            {
                "replies": [
                    {"replaceAllText": {"occurrencesChanged": 2}},
                    {"replaceAllText": {}},
                ]
            },
        ),
    )
    outputs = dict(
        await _run(
            GoogleSlidesReplaceAllTextBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            replacements={"{{client}}": "Acme Corp", "{{date}}": "1 October"},
            match_case=False,
            slide_ids=[SLIDE_LINK, "p"],
        )
    )
    first, second = http.calls[0][2]["requests"]
    assert first == {
        "replaceAllText": {
            "containsText": {"text": "{{client}}", "matchCase": False},
            "replaceText": "Acme Corp",
            "pageObjectIds": ["g2f3c4d5e6_0_0", "p"],
        }
    }
    assert second["replaceAllText"]["containsText"]["text"] == "{{date}}"
    assert outputs["occurrences"] == {"{{client}}": 2, "{{date}}": 0}
    assert outputs["occurrences_changed"] == 2


@pytest.mark.asyncio
async def test_replace_all_text_matches_case_on_every_slide_by_default(slides_api):
    http = slides_api(slides_edit, (200, {"replies": [{}]}))
    await _run(
        GoogleSlidesReplaceAllTextBlock(),
        presentation=TEST_PICKED_PRESENTATION,
        replacements={"{{client}}": "Acme Corp"},
    )
    (request,) = http.calls[0][2]["requests"]
    assert request["replaceAllText"]["containsText"]["matchCase"] is True
    assert "pageObjectIds" not in request["replaceAllText"]


@pytest.mark.asyncio
@pytest.mark.parametrize("replacements", [{}, {"": "Acme Corp"}])
async def test_replace_all_text_rejects_empty_find_text(slides_api, replacements):
    http = slides_api(slides_edit)
    with pytest.raises(BlockInputError, match="text to find"):
        await _run(
            GoogleSlidesReplaceAllTextBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            replacements=replacements,
        )
    assert http.calls == []


def _create_reply(slide_id: str) -> tuple[int, dict[str, Any]]:
    return 200, {"replies": [{"createSlide": {"objectId": slide_id}}]}


@pytest.mark.asyncio
async def test_add_slide_fills_the_title_layouts_placeholders(slides_api):
    title_slide = {
        "objectId": "s1",
        "pageElements": [
            placeholder_shape("centered", "CENTERED_TITLE"),
            placeholder_shape("subtitle", "SUBTITLE"),
        ],
    }
    http = slides_api(
        slides_create,
        _create_reply("s1"),
        (200, title_slide),
        (200, {"replies": [{}, {}]}),
    )
    outputs = await _run(
        GoogleSlidesAddSlideBlock(),
        presentation=TEST_PICKED_PRESENTATION,
        layout="title",
        title="Q4 plan",
        body="Draft for review",
        index=0,
    )
    assert http.calls[0][2] == {
        "requests": [
            {
                "createSlide": {
                    "slideLayoutReference": {"predefinedLayout": "TITLE"},
                    "insertionIndex": 0,
                }
            }
        ]
    }
    assert urlparse(http.calls[1][1]).path.endswith("/pages/s1")
    assert http.calls[2][2] == {
        "requests": [
            {"insertText": {"objectId": "centered", "text": "Q4 plan"}},
            {"insertText": {"objectId": "subtitle", "text": "Draft for review"}},
        ]
    }
    assert outputs[0] == ("slide_id", "s1")


@pytest.mark.asyncio
async def test_add_slide_without_text_only_creates_it_at_the_end(slides_api):
    http = slides_api(slides_create, _create_reply("s2"))
    outputs = await _run(
        GoogleSlidesAddSlideBlock(),
        presentation=TEST_PICKED_PRESENTATION,
        layout="blank",
    )
    assert len(http.calls) == 1
    create = http.calls[0][2]["requests"][0]["createSlide"]
    assert create == {"slideLayoutReference": {"predefinedLayout": "BLANK"}}
    assert outputs[0] == ("slide_id", "s2")


@pytest.mark.asyncio
async def test_add_slide_removes_the_slide_when_the_layout_has_no_body(slides_api):
    header = {"objectId": "s3", "pageElements": [placeholder_shape("t3", "TITLE")]}
    http = slides_api(
        slides_create, _create_reply("s3"), (200, header), (200, {"replies": [{}]})
    )
    with pytest.raises(BlockInputError, match="no body placeholder"):
        await _run(
            GoogleSlidesAddSlideBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            layout="section_header",
            title="Part 2",
            body="Details",
        )
    assert http.calls[2][2] == {"requests": [{"deleteObject": {"objectId": "s3"}}]}


@pytest.mark.asyncio
async def test_add_slide_rejects_text_on_the_blank_layout_before_calling_google(
    slides_api,
):
    http = slides_api(slides_create)
    with pytest.raises(BlockInputError, match="blank layout"):
        await _run(
            GoogleSlidesAddSlideBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            layout="blank",
            title="Hello",
        )
    assert http.calls == []


@pytest.mark.asyncio
async def test_add_slide_explains_missing_layouts(slides_api):
    slides_api(
        slides_create,
        _error(400, "Invalid requests[0].createSlide: The layout could not be found."),
    )
    with pytest.raises(BlockExecutionError, match="PowerPoint"):
        await _run(
            GoogleSlidesAddSlideBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            layout="big_number",
        )


def test_find_placeholder_prefers_type_order_then_lowest_index():
    page = {
        "pageElements": [
            placeholder_shape("right", "BODY", index=1),
            placeholder_shape("subtitle", "SUBTITLE"),
            placeholder_shape("left", "BODY", index=0),
            {"objectId": "box", "shape": {"shapeType": "TEXT_BOX"}},
        ]
    }
    assert find_placeholder(page, ("BODY", "SUBTITLE")) == "left"
    assert find_placeholder(page, ("SUBTITLE", "BODY")) == "subtitle"
    assert find_placeholder(page, ("TITLE", "CENTERED_TITLE")) is None


_DELETE_NOTES = {
    "deleteText": {"objectId": "g2f3c4d5e6_0_3", "textRange": {"type": "ALL"}}
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "slide, notes, expected",
    [
        (
            TEST_REVENUE_SLIDE,
            "New notes",
            [
                _DELETE_NOTES,
                {
                    "insertText": {
                        "objectId": "g2f3c4d5e6_0_3",
                        "insertionIndex": 0,
                        "text": "New notes",
                    }
                },
            ],
        ),
        (
            TEST_NEW_SLIDE,
            "First notes",
            [
                {
                    "insertText": {
                        "objectId": "SLIDES_API1712345678_3",
                        "insertionIndex": 0,
                        "text": "First notes",
                    }
                }
            ],
        ),
        (TEST_REVENUE_SLIDE, "", [_DELETE_NOTES]),
    ],
    ids=["replace", "first-notes", "clear"],
)
async def test_set_speaker_notes_requests(slides_api, slide, notes, expected):
    http = slides_api(slides_edit, (200, slide), (200, {"replies": [{}, {}]}))
    await _run(
        GoogleSlidesSetSpeakerNotesBlock(),
        presentation=TEST_PICKED_PRESENTATION,
        slide_id=slide["objectId"],
        notes=notes,
    )
    assert http.calls[1][2] == {"requests": expected}


@pytest.mark.asyncio
async def test_clearing_notes_that_are_already_empty_changes_nothing(slides_api):
    http = slides_api(slides_edit, (200, TEST_NEW_SLIDE))
    await _run(
        GoogleSlidesSetSpeakerNotesBlock(),
        presentation=TEST_PICKED_PRESENTATION,
        slide_id=TEST_NEW_SLIDE["objectId"],
        notes="",
    )
    assert [method for method, _, _ in http.calls] == ["GET"]


@pytest.mark.asyncio
async def test_set_speaker_notes_refuses_pages_that_are_not_slides(slides_api):
    slides_api(slides_edit, (200, {"objectId": "layout1", "pageType": "LAYOUT"}))
    with pytest.raises(BlockInputError, match="not a slide"):
        await _run(
            GoogleSlidesSetSpeakerNotesBlock(),
            presentation=TEST_PICKED_PRESENTATION,
            slide_id="layout1",
            notes="Hi",
        )


@pytest.mark.parametrize(
    "value, expected",
    [
        ("g2f3c4d5e6_0_0", "g2f3c4d5e6_0_0"),
        ("  p  ", "p"),
        (SLIDE_LINK, "g2f3c4d5e6_0_0"),
        (
            "https://docs.google.com/presentation/d/abc/edit?slide=id.SLIDES_API1_0#slide=id.SLIDES_API1_0",
            "SLIDES_API1_0",
        ),
    ],
)
def test_require_slide_id_accepts_ids_and_slide_links(value: str, expected: str):
    assert require_slide_id(value, "block", "id") == expected


@pytest.mark.parametrize(
    "value", ["", "https://docs.google.com/presentation/d/abc/edit"]
)
def test_require_slide_id_rejects_links_without_a_slide(value: str):
    with pytest.raises(BlockInputError, match="slide ID"):
        require_slide_id(value, "block", "id")


def test_require_presentation_rejects_other_files():
    with pytest.raises(BlockInputError, match="Pick a Google Slides"):
        require_presentation(None, "block", "id")
    pptx = GoogleDriveFile.model_validate(
        {
            "id": "x1",
            "name": "deck.pptx",
            "mimeType": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
    )
    with pytest.raises(BlockInputError, match="Save as Google Slides"):
        require_presentation(pptx, "block", "id")
    untyped = GoogleDriveFile(id="x2")
    assert require_presentation(untyped, "block", "id") is untyped


@pytest.mark.parametrize(
    "status, reason, expected",
    [
        (404, "Requested entity was not found.", "couldn't find that presentation"),
        (403, "Request had insufficient authentication scopes.", "Reconnect Google"),
        (403, "The caller does not have permission", "denied access"),
        (400, "Invalid requests[0].createSlide", "API error 400: Invalid requests"),
    ],
)
def test_slides_error_messages(status: int, reason: str, expected: str):
    content = json.dumps({"error": {"code": status, "message": reason}}).encode()
    exc = HttpError(httplib2.Response({"status": status}), content)
    assert expected in str(slides_error(exc, "block", "id"))


def test_bad_request_hint_is_only_added_to_400s():
    def error(status: int) -> HttpError:
        content = json.dumps({"error": {"code": status, "message": "x"}}).encode()
        return HttpError(httplib2.Response({"status": status}), content)

    assert "Hint" in str(slides_error(error(400), "b", "i", "Hint."))
    assert "Hint" not in str(slides_error(error(500), "b", "i", "Hint."))


def test_presentation_output_chains_into_picker_fields():
    """An output presentation must feed GoogleDriveFile inputs with its credentials."""
    serialized = to_dict(presentation_file("deck1", "Deck", "cred-123"))
    assert serialized["_credentials_id"] == "cred-123"
    assert serialized["url"] == "https://docs.google.com/presentation/d/deck1/edit"
    picked = GoogleDriveFile.model_validate(serialized)
    assert (picked.id, picked.credentials_id) == ("deck1", "cred-123")
    assert require_presentation(picked, "block", "id") is picked


@pytest.mark.parametrize(
    "block_cls, scopes",
    [
        (GoogleSlidesReadPresentationBlock, {DRIVE_FILE_SCOPE}),
        (GoogleSlidesGetSlideBlock, {DRIVE_FILE_SCOPE}),
        (GoogleSlidesGetSlideThumbnailBlock, {DRIVE_FILE_SCOPE}),
        (GoogleSlidesCreatePresentationBlock, {DRIVE_FILE_SCOPE}),
        (GoogleSlidesBatchUpdateBlock, {DRIVE_FILE_SCOPE, PRESENTATIONS_SCOPE}),
        (GoogleSlidesReplaceAllTextBlock, {DRIVE_FILE_SCOPE, PRESENTATIONS_SCOPE}),
        (GoogleSlidesAddSlideBlock, {DRIVE_FILE_SCOPE, PRESENTATIONS_SCOPE}),
        (GoogleSlidesSetSpeakerNotesBlock, {DRIVE_FILE_SCOPE, PRESENTATIONS_SCOPE}),
    ],
)
def test_blocks_ask_for_the_approved_scopes_and_are_reversible(block_cls, scopes):
    (field,) = block_cls.Input.get_credentials_fields_info().values()
    assert field.required_scopes == scopes
    assert block_cls().is_irreversible_action is False


@pytest.mark.parametrize(
    "root, mask",
    [
        ("Presentation", slides_read.PRESENTATION_FIELDS),
        ("Presentation", "presentationId,title"),
    ],
)
def test_field_masks_match_the_slides_schema(root: str, mask: str):
    """A typo in a field mask only fails against the live API, so check the
    masks against the discovery document the client ships with."""
    path = (
        Path(googleapiclient.__file__).parent
        / "discovery_cache/documents/slides.v1.json"
    )
    schemas = json.loads(path.read_text(encoding="utf-8"))["schemas"]
    for field_path in _mask_paths(mask):
        schema = schemas[root]
        for name in field_path:
            assert name in schema["properties"], f"{'.'.join(field_path)} in {mask}"
            prop = schema["properties"][name]
            ref = prop.get("$ref") or prop.get("items", {}).get("$ref")
            schema = schemas[ref] if ref else {"properties": {}}


def _mask_paths(mask: str) -> list[list[str]]:
    """Expand a field mask like 'a,b(c,d(e))' into [[a], [b, c], [b, d, e]]."""
    paths: list[list[str]] = []
    stack: list[list[str]] = [[]]
    name = ""
    for char in mask + ",":
        if char == "(":
            stack.append(stack[-1] + name.split("."))
            name = ""
        elif char in ",)":
            if name:
                paths.append(stack[-1] + name.split("."))
            name = ""
            if char == ")":
                stack.pop()
        else:
            name += char.strip()
    return paths


def test_mask_paths_expands_nested_selections():
    assert _mask_paths("a,b(c,d.e(f)),g") == [
        ["a"],
        ["b", "c"],
        ["b", "d", "e", "f"],
        ["g"],
    ]
