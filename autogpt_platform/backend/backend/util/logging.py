import logging

from backend.util.settings import AppEnvironment, BehaveAs, Settings

settings = Settings()


def configure_logging():
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
