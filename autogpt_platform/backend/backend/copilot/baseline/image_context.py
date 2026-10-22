"""Keep the baseline's image input bounded between tool rounds."""

from typing import Any

MAX_IMAGE_CONTEXT_COUNT = 10
# Inline image URLs are ASCII. Leave space for text, tools and JSON overhead.
MAX_IMAGE_CONTEXT_CHARS = 20 * 1024 * 1024
IMAGE_OMITTED_MESSAGE = (
    "[Image omitted from the current model input to stay within the image budget. "
    "Do not infer its contents. Read the referenced file again when needed; "
    "inspect fewer images at a time or resize them if necessary.]"
)


def prune_image_context(messages: list[dict[str, Any]]) -> None:
    """Keep recent images, replacing older/oversized blocks with reread guidance.

    Only the live provider context changes. Text, tool pairs and the durable
    transcript retain their original content and references.
    """
    remaining_count = MAX_IMAGE_CONTEXT_COUNT
    remaining_chars = MAX_IMAGE_CONTEXT_CHARS
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        bounded = content
        for index in range(len(content) - 1, -1, -1):
            block = content[index]
            if block.get("type") != "image_url":
                continue
            size = len(block["image_url"]["url"])
            if remaining_count > 0 and size <= remaining_chars:
                remaining_count -= 1
                remaining_chars -= size
                continue
            if bounded is content:
                bounded = list(content)
            bounded[index] = {"type": "text", "text": IMAGE_OMITTED_MESSAGE}
        if bounded is not content:
            message["content"] = bounded
