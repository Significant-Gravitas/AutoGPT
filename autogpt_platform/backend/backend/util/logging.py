import logging
from collections.abc import Mapping, Sequence
from typing import Any

from backend.util.settings import AppEnvironment, BehaveAs, Settings
from backend.util.url_redaction import CredentialHeaderFilter, RequestURLRedactionFilter

settings = Settings()

_request_url_redaction = RequestURLRedactionFilter()
_credential_header_redaction = CredentialHeaderFilter()


def configure_logging():
    import autogpt_libs.logging.config

    if not is_structured_logging_enabled():
        autogpt_libs.logging.config.configure_logging(force_cloud_logging=False)
    else:
        autogpt_libs.logging.config.configure_logging(force_cloud_logging=True)

    # Silence httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # These log request URLs we don't format: uvicorn the path and query of
    # every request (the access log, and a line per WebSocket handshake),
    # autogpt_libs the URL of every request an admin makes as another user, and
    # websockets, at DEBUG, the target of each handshake: a server's on
    # uvicorn's error log, and a client's, or a server's outside uvicorn, on its
    # own. A parent logger's filters never see records propagated from its
    # children, so the filter goes on each logger directly.
    for name in (
        "uvicorn.access",
        "uvicorn.error",
        "autogpt_libs.auth.dependencies",
        "websockets.client",
        "websockets.server",
    ):
        logging.getLogger(name).addFilter(_request_url_redaction)
    # websockets logs each handshake's headers there too, the session cookie
    # and Authorization included.
    for name in ("uvicorn.error", "websockets.client", "websockets.server"):
        logging.getLogger(name).addFilter(_credential_header_redaction)


def describe_validation_errors(errors: Sequence[Mapping[str, Any]]) -> str:
    """Where and why validation failed, for the logs, without what failed it.

    Each pydantic error carries the ``input`` that failed, which for a missing
    field is the whole object around it, such as a request body holding an API
    key; its message can quote the input too.
    """
    locations = "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc') or ())} "
        f"({error.get('type')})"
        for error in errors
    )
    count = len(errors)
    return f"{count} validation error{'' if count == 1 else 's'}: {locations}"


def is_structured_logging_enabled() -> bool:
    """Check if structured logging (cloud logging) is enabled."""
    return not (
        settings.config.behave_as == BehaveAs.LOCAL
        or settings.config.app_env == AppEnvironment.LOCAL
    )


class TruncatedLogger:
    def __init__(
        self,
        logger: logging.Logger,
        prefix: str = "",
        metadata: dict | None = None,
        max_length: int = 1000,
    ):
        self.logger = logger
        self.metadata = metadata or {}
        self.max_length = max_length
        self.prefix = prefix

    def info(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        self.logger.info(msg, extra=self._get_metadata(**extra))

    def warning(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        self.logger.warning(msg, extra=self._get_metadata(**extra))

    def error(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        self.logger.error(msg, extra=self._get_metadata(**extra))

    def debug(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        self.logger.debug(msg, extra=self._get_metadata(**extra))

    def exception(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        self.logger.exception(msg, extra=self._get_metadata(**extra))

    def _get_metadata(self, **extra):
        metadata = {**self.metadata, **extra}
        return {"json_fields": metadata} if metadata else {}

    def _wrap(self, msg: str, **extra):
        extra_msg = str(extra or "")
        text = f"{self.prefix} {msg} {extra_msg}"
        if len(text) > self.max_length:
            half = (self.max_length - 3) // 2
            text = text[:half] + "..." + text[-half:]
        return text


class PrefixFilter(logging.Filter):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def filter(self, record):
        record.msg = f"{self.prefix} {record.msg}"
        return True
