from typing import Any
from unittest.mock import MagicMock, call

import pytest
from tiktoken import encoding_for_model

from backend.util import prompt


@pytest.mark.parametrize("payload_size", [64, 4 * 1024 * 1024])
@pytest.mark.parametrize("image_count", [1, 10])
@pytest.mark.parametrize(
    "image_type,container",
    [
        ("image_url", "message"),
        ("image", "message"),
        ("input_image", "message"),
        ("image", "tool_result"),
        ("input_image", "function_call_output"),
    ],
)
def test_image_estimate_does_not_tokenize_pixels(
    monkeypatch: pytest.MonkeyPatch,
    payload_size: int,
    image_count: int,
    image_type: str,
    container: str,
) -> None:
    encoding = encoding_for_model("gpt-4o")

    def encode_text(text: str) -> list[int]:
        if "data:image/" in text or "YQ==" in text:
            raise AssertionError("Image payload reached the text tokenizer")
        return encoding.encode(text)

    guarded_encoding = MagicMock()
    guarded_encoding.encode.side_effect = encode_text
    monkeypatch.setattr(prompt, "encoding_for_model", lambda _: guarded_encoding)
    caption = "Image returned by read_workspace_file:"
    payload = "YQ==" * (payload_size // 4)
    data_url = "data:image/png;base64," + payload
    image_block = {
        "image_url": {"type": "image_url", "image_url": {"url": data_url}},
        "image": {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": payload},
        },
        "input_image": {"type": "input_image", "image_url": data_url},
    }[image_type]
    content = [
        {
            "type": "input_text" if image_type == "input_image" else "text",
            "text": caption,
        },
        *[image_block for _ in range(image_count)],
    ]
    expected_text = ["Inspect the pages", caption]
    message: dict[str, Any] = {"role": "user", "content": content}
    if container == "tool_result":
        message["content"] = [
            {"type": "tool_result", "tool_use_id": "image-read", "content": content}
        ]
        expected_text.insert(1, "image-read")
    elif container == "function_call_output":
        message = {
            "type": "function_call_output",
            "call_id": "image-read",
            "output": content,
        }
        expected_text.append("image-read")
    messages = [{"role": "user", "content": "Inspect the pages"}, message]

    tokens = prompt.estimate_token_count(messages)

    assert guarded_encoding.encode.call_args_list == [
        call(text) for text in expected_text
    ]
    assert tokens == (
        6
        + sum(len(encoding.encode(text)) for text in expected_text)
        + image_count * 4096
    )


@pytest.mark.parametrize(
    "block_type,text_field",
    [("output_text", "text"), ("text", "content"), ("legacy", "content")],
)
def test_structured_text_preserves_token_count(
    block_type: str, text_field: str
) -> None:
    text = "The image shows three blue squares."
    structured = {
        "role": "assistant",
        "content": [{"type": block_type, text_field: text}],
    }
    plain = {"role": "assistant", "content": text}

    assert prompt.estimate_token_count([structured]) == prompt.estimate_token_count(
        [plain]
    )
