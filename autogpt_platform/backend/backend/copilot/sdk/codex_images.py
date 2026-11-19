from backend.copilot.tool_image import IMAGE_MIME_TYPES


def image_urls_from_content(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        _image_url(block.get("source"))
        for block in value
        if isinstance(block, dict) and block.get("type") == "image"
    ]


def normalize_content(value: object, image_urls: list[str]) -> object:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    normalized: list[object] = []
    for block in value:
        if not isinstance(block, dict) or block.get("type") == "thinking":
            continue
        if block.get("type") == "image":
            image_urls.append(_image_url(block.get("source")))
            normalized.append(
                {
                    "type": "text",
                    "text": f"[Image {len(image_urls)} attached after this transcript]",
                }
            )
            continue
        normalized.append(
            {
                key: normalize_content(item, image_urls) if key == "content" else item
                for key, item in block.items()
                if key not in {"cache_control", "signature"}
            }
        )
    return normalized


def _image_url(source: object) -> str:
    if isinstance(source, dict):
        if source.get("type") == "base64":
            mime, data = source.get("media_type"), source.get("data")
            if (
                isinstance(mime, str)
                and mime in IMAGE_MIME_TYPES
                and isinstance(data, str)
                and data
            ):
                return f"data:{mime};base64,{data}"
    raise ValueError(
        "Image not shown: provide PNG/JPEG/GIF/WebP image bytes through a file read "
        "or attachment; URL image sources are not supported."
    )
