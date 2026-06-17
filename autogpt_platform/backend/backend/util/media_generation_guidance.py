import re

IMAGE_GENERATION_LEADERBOARD_URL = "https://arena.ai/leaderboard/text-to-image"
VIDEO_GENERATION_LEADERBOARD_URL = "https://arena.ai/leaderboard/text-to-video"

IMAGE_GENERATION_MODEL_SELECTION_GUIDANCE = (
    "When picking a model automatically, prefer the top-ranked text-to-image "
    f"model on {IMAGE_GENERATION_LEADERBOARD_URL} unless a specific model is "
    "requested."
)

VIDEO_GENERATION_MODEL_SELECTION_GUIDANCE = (
    "When picking a model automatically, prefer the top-ranked text-to-video "
    f"model on {VIDEO_GENERATION_LEADERBOARD_URL} unless a specific model is "
    "requested."
)

IMAGE_GENERATION_FALLBACK_GUIDANCE = (
    "If the user did not explicitly request this specific model, try another "
    "image generation model. If they did request it, tell them the selected "
    "model appears to be down or unavailable."
)

VIDEO_GENERATION_FALLBACK_GUIDANCE = (
    "If the user did not explicitly request this specific model or provider, try "
    "another video generation model or block. If they did request it, tell them "
    "the selected model appears to be down or unavailable."
)

_FALLBACK_ERROR_MARKERS = (
    "unavailable",
    "timed out",
    "timeout",
    "rate limit",
    "rate-limited",
    "overloaded",
    "server error",
    "internal error",
    "bad gateway",
    "gateway timeout",
    "maximum polling attempts",
    "max polling attempts",
    "no output",
    "empty result",
    "no project id",
    "no clip id",
    "missing required data in submission response",
    "invalid response format",
    "unable to process",
    "no valid video url",
)

# Hard no-fallback markers: switching models cannot help, even with transient
# wording in the same message.
_HARD_NO_FALLBACK_ERROR_MARKERS = (
    "moderation",
    "moderated",
    "flagged",
    "sensitive",
    "nsfw",
    "policy",
    "safety",
    "unsafe",
    "prohibited content",
    "content violation",
    "content filter",
    "invalid api key",
    "unauthorized",
    "forbidden",
    "authentication",
    "credential",
    "permission denied",
    "quota exceeded",
    "billing",
    "insufficient credits",
    "insufficient quota",
    "insufficient permissions",
    "insufficient funds",
    "insufficient balance",
)

# Soft no-fallback markers: suppress fallback only when no explicit fallback
# marker is present in the same message.
_SOFT_NO_FALLBACK_ERROR_MARKERS = (
    "invalid prompt",
    "invalid input",
    "invalid request",
    "validation error",
    "validation failed",
)

_FALLBACK_STATUS_RE = re.compile(r"\b(?:408|429|500|502|503|504)\b")
_NO_FALLBACK_STATUS_RE = re.compile(r"\b(?:400|401|402|403|404)\b")


def image_generation_failure_message(message: str) -> str:
    return _generation_failure_message(
        message=message,
        default_message="Image generation failed.",
        fallback_guidance=IMAGE_GENERATION_FALLBACK_GUIDANCE,
    )


def video_generation_failure_message(message: str) -> str:
    return _generation_failure_message(
        message=message,
        default_message="Video generation failed.",
        fallback_guidance=VIDEO_GENERATION_FALLBACK_GUIDANCE,
    )


def _generation_failure_message(
    *, message: str, default_message: str, fallback_guidance: str
) -> str:
    failure_message = _as_sentence(message, default_message)
    if fallback_guidance in failure_message:
        return failure_message
    if _should_suggest_fallback(failure_message):
        return f"{failure_message} {fallback_guidance}"
    return failure_message


def _should_suggest_fallback(message: str) -> bool:
    normalized_message = message.casefold()
    if any(marker in normalized_message for marker in _HARD_NO_FALLBACK_ERROR_MARKERS):
        return False
    # Explicit fallback markers win over soft no-fallback markers + status codes.
    if any(marker in normalized_message for marker in _FALLBACK_ERROR_MARKERS):
        return True
    if any(marker in normalized_message for marker in _SOFT_NO_FALLBACK_ERROR_MARKERS):
        return False
    if _NO_FALLBACK_STATUS_RE.search(normalized_message):
        return False
    return bool(_FALLBACK_STATUS_RE.search(normalized_message))


def _as_sentence(message: str, default_message: str) -> str:
    stripped_message = message.strip()
    if not stripped_message:
        return default_message
    if stripped_message.endswith((".", "!", "?")):
        return stripped_message
    return f"{stripped_message}."


def response_detail(response: dict) -> str | None:
    """Pick the first useful human-readable detail field from a provider response."""
    for key in ("error", "message", "detail", "status"):
        value = response.get(key)
        if not value:
            continue
        if isinstance(value, dict):
            nested = response_detail(value)
            if nested:
                return nested
            continue
        return str(value)
    return None
