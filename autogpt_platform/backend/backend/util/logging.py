from logging import Logger

from backend.util.settings import AppEnvironment, BehaveAs, Settings

settings = Settings()


def configure_logging():
    import logging

    import autogpt_libs.logging.config

    if (
        settings.config.behave_as == BehaveAs.LOCAL
        or settings.config.app_env == AppEnvironment.LOCAL
    ):
        autogpt_libs.logging.config.configure_logging(force_cloud_logging=False)
    else:
        autogpt_libs.logging.config.configure_logging(force_cloud_logging=True)

    # Silence httpx logger
    logging.getLogger("httpx").setLevel(logging.WARNING)


class TruncatedLogger:
    def __init__(
        self,
        logger: Logger,
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
        if len(extra_msg) > 1000:
            extra_msg = extra_msg[:1000] + "..."
        return f"{self.prefix} {msg} {extra_msg}"
