import hashlib

from backend.util.settings import Settings

settings = Settings()


def copilot_trace_metadata(base_prompt: str | None = None) -> dict[str, str]:
    """Identify the deployment and resolved base prompt, before turn supplements."""
    metadata: dict[str, str] = {}
    if release := settings.config.sentry_release.strip():
        metadata["release"] = release
    if base_prompt is not None:
        metadata["base_prompt_sha256"] = hashlib.sha256(
            base_prompt.encode("utf-8")
        ).hexdigest()
    return metadata
