import io
import json

import pytest
from langfuse import get_client
from PIL import Image

from backend.copilot.google_data_redaction import redact_google_data
from backend.copilot.google_data_redaction_test import langfuse_spans
from backend.copilot.tool_image import prepare_image

__all__ = ["langfuse_spans"]


@pytest.mark.parametrize("format", ["anthropic", "mcp", "openai", "responses"])
def test_image_pixels_stay_out_of_traces_without_changing_model_input(
    format, langfuse_spans
):
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(output, format="PNG")
    image = prepare_image(output.getvalue())
    blocks = {
        "anthropic": image.anthropic(),
        "mcp": image.mcp(),
        "openai": image.openai(),
        "responses": {
            "type": "input_image",
            "image_url": image.openai()["image_url"]["url"],
        },
    }
    payload = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Inspect this page"},
            blocks[format],
        ],
    }
    before = json.dumps(payload)
    generation = get_client().start_observation(
        name="image-inspection", as_type="generation", input=payload
    )
    generation.end()
    traced = json.dumps(
        [dict(span.attributes or {}) for span in langfuse_spans.get_finished_spans()]
    )
    assert image.data_base64 not in traced
    assert image.data_base64 not in json.dumps(redact_google_data(json.loads(before)))
    assert "Inspect this page" in traced
    assert json.dumps(payload) == before
    assert image.data_base64 in before


@pytest.mark.parametrize("mixed", [False, True])
def test_nested_tool_result_pixels_are_redacted(mixed):
    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "private-image-bytes",
        },
    }
    if mixed:
        image_block["data"] = "other-private-image-bytes"
    payload = {
        "type": "tool_result",
        "tool_use_id": "call-image",
        "content": [
            {"type": "text", "text": "Viewed page"},
            image_block,
        ],
    }
    before = json.dumps(payload)
    traced = redact_google_data(payload)
    assert traced["content"][0]["text"] == "Viewed page"
    assert "private-image-bytes" not in json.dumps(traced)
    assert traced["tool_use_id"] == "call-image"
    assert json.dumps(payload) == before


def test_non_image_data_stays_readable():
    payload = {"type": "text", "data": "public data", "source": {"data": "citation"}}
    assert redact_google_data(payload) == payload
