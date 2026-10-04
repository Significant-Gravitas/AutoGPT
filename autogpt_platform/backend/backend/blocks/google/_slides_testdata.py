"""Shared test fixtures for the Google Slides blocks.

Kept in their own module so block files never import test data from each other.
"""

from typing import Any

from ._auth import TEST_CREDENTIALS
from ._slides_api import PRESENTATION_MIME_TYPE, presentation_file

TEST_PRESENTATION_ID = "1QnV2p8b4TmZ7xYc3dRk9LsWq0aEfGhJiKlMnOpQrStU"
TEST_PICKED_PRESENTATION = {
    "id": TEST_PRESENTATION_ID,
    "name": "Q3 Business Review",
    "mimeType": PRESENTATION_MIME_TYPE,
}
TEST_PRESENTATION_FILE = presentation_file(
    TEST_PRESENTATION_ID, "Q3 Business Review", TEST_CREDENTIALS.id
)


def text_content(text: str) -> dict[str, Any]:
    """A shape's text as the Slides API returns it."""
    return {
        "textElements": [
            {"paragraphMarker": {}},
            {"textRun": {"content": f"{text}\n"}},
        ]
    }


def placeholder_shape(
    object_id: str, placeholder: str, text: str = "", index: int = 0
) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "shapeType": "TEXT_BOX",
        "placeholder": {"type": placeholder, "index": index},
    }
    if text:
        shape["text"] = text_content(text)
    return {"objectId": object_id, "shape": shape}


def notes_properties(object_id: str, text: str = "") -> dict[str, Any]:
    """A slide's slideProperties with its notes page. Without text the notes
    shape is left out, as Google does for slides that never had notes."""
    elements = [placeholder_shape(object_id, "BODY", text)] if text else []
    return {
        "notesPage": {
            "notesProperties": {"speakerNotesObjectId": object_id},
            "pageElements": elements,
        }
    }


TEST_REVENUE_SLIDE = {
    "objectId": "g2f3c4d5e6_0_0",
    "pageType": "SLIDE",
    "pageElements": [
        placeholder_shape("g2f3c4d5e6_0_1", "TITLE", "Revenue"),
        {
            "objectId": "g2f3c4d5e6_0_2",
            "table": {
                "rows": 2,
                "columns": 2,
                "tableRows": [
                    {"tableCells": [{"text": text_content(t)} for t in row]}
                    for row in (("Region", "Revenue"), ("EMEA", "$1.2M"))
                ],
            },
        },
    ],
    "slideProperties": notes_properties("g2f3c4d5e6_0_3", "Call out the EMEA launch."),
}
TEST_PRESENTATION = {
    "presentationId": TEST_PRESENTATION_ID,
    "title": "Q3 Business Review",
    "slides": [
        {
            "objectId": "p",
            "pageType": "SLIDE",
            "pageElements": [
                placeholder_shape("i0", "CENTERED_TITLE", "Q3 Business Review"),
                placeholder_shape("i1", "SUBTITLE", "Finance team"),
            ],
            "slideProperties": notes_properties("i3", "Welcome everyone."),
        },
        TEST_REVENUE_SLIDE,
    ],
}
TEST_NEW_SLIDE = {
    "objectId": "SLIDES_API1712345678_0",
    "pageType": "SLIDE",
    "pageElements": [
        placeholder_shape("SLIDES_API1712345678_1", "TITLE"),
        placeholder_shape("SLIDES_API1712345678_2", "BODY"),
    ],
    "slideProperties": notes_properties("SLIDES_API1712345678_3"),
}
