from unittest.mock import MagicMock, call

import pytest
from tiktoken import encoding_for_model

from backend.util import prompt


@pytest.mark.parametrize("payload_size", [64, 4 * 1024 * 1024])
@pytest.mark.parametrize("image_count", [1, 10])
def test_image_estimate_does_not_tokenize_pixels(
    monkeypatch: pytest.MonkeyPatch, payload_size: int, image_count: int
) -> None:
    encoding = encoding_for_model("gpt-4o")

    def encode_text(text: str) -> list[int]:
        if "data:image/" in text:
            raise AssertionError("Image payload reached the text tokenizer")
        return encoding.encode(text)

    guarded_encoding = MagicMock()
    guarded_encoding.encode.side_effect = encode_text
    monkeypatch.setattr(prompt, "encoding_for_model", lambda _: guarded_encoding)
    caption = "Image returned by read_workspace_file:"
    messages = [
        {"role": "user", "content": "Inspect the pages"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": caption},
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + "YQ==" * (payload_size // 4)
                        },
                    }
                    for _ in range(image_count)
                ],
            ],
        },
    ]

    tokens = prompt.estimate_token_count(messages)

    assert guarded_encoding.encode.call_args_list == [
        call("Inspect the pages"),
        call(caption),
    ]
    assert tokens == (
        6
        + len(encoding.encode("Inspect the pages"))
        + len(encoding.encode(caption))
        + image_count * 4096
    )
