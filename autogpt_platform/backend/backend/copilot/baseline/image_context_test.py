from typing import Any

import pytest

from backend.copilot.baseline import image_context


def test_keeps_the_most_recent_images_and_preserves_text_and_tool_pairs():
    images = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,image{index}"},
        }
        for index in range(12)
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": [{"type": "text", "text": "Initial"}, *images[:6]]},
        {"role": "assistant", "tool_calls": [{"id": "read"}]},
        {"role": "tool", "tool_call_id": "read", "content": "workspace://page.png"},
        {"role": "user", "content": [*images[6:], {"type": "text", "text": "Latest"}]},
    ]
    tool_pair = messages[2:4].copy()

    image_context.prune_image_context(messages)

    assert messages[0] == {"role": "system", "content": "System prompt"}
    assert messages[1]["content"] == [
        {"type": "text", "text": "Initial"},
        *[{"type": "text", "text": image_context.IMAGE_OMITTED_MESSAGE}] * 2,
        *images[2:6],
    ]
    assert messages[2:4] == tool_pair
    assert messages[-1]["content"] == [
        *images[6:],
        {"type": "text", "text": "Latest"},
    ]
    assert all(image["type"] == "image_url" for image in images)
    once = [message.copy() for message in messages]
    image_context.prune_image_context(messages)
    assert messages == once


@pytest.mark.parametrize("overflow", [0, 1])
def test_enforces_the_encoded_size_boundary_without_counting_text(
    monkeypatch, overflow
):
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
    }
    size = len(image["image_url"]["url"])
    monkeypatch.setattr(image_context, "MAX_IMAGE_CONTEXT_CHARS", 2 * size - overflow)
    text = {"type": "text", "text": "Keep this reference and analysis." * 100}
    messages = [{"role": "user", "content": [image, text, image]}]

    image_context.prune_image_context(messages)

    assert messages[0]["content"][1] == text
    assert messages[0]["content"][2] == image
    first = messages[0]["content"][0]
    assert first["type"] == ("text" if overflow else "image_url")
    if overflow:
        assert "omitted" in first["text"]
        assert "read" in first["text"].lower()
        assert "resize" in first["text"].lower()


def test_a_single_oversized_image_gets_guidance_and_does_not_use_the_budget(
    monkeypatch,
):
    small = {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}
    big = {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWFhYQ=="}}
    monkeypatch.setattr(
        image_context, "MAX_IMAGE_CONTEXT_CHARS", len(small["image_url"]["url"])
    )
    messages = [
        {"role": "user", "content": [small]},
        {"role": "user", "content": [big]},
    ]

    image_context.prune_image_context(messages)

    assert messages[0]["content"] == [small]
    assert messages[1]["content"] == [
        {"type": "text", "text": image_context.IMAGE_OMITTED_MESSAGE}
    ]


@pytest.mark.parametrize(
    "content", [None, "", "Only text", [{"type": "text", "text": "Hi"}]]
)
def test_no_images_preserves_the_message(content):
    messages = [{"role": "user", "content": content}]
    image_context.prune_image_context(messages)
    assert messages == [{"role": "user", "content": content}]
