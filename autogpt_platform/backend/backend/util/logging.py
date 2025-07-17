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
        truncated_extra = self._truncate_large_data(extra)
        truncated_metadata = self._truncate_large_data(self.metadata)
        self.logger.info(
            msg, extra={"json_fields": {**truncated_metadata, **truncated_extra}}
        )

    def warning(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        truncated_extra = self._truncate_large_data(extra)
        truncated_metadata = self._truncate_large_data(self.metadata)
        self.logger.warning(
            msg, extra={"json_fields": {**truncated_metadata, **truncated_extra}}
        )

    def error(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        truncated_extra = self._truncate_large_data(extra)
        truncated_metadata = self._truncate_large_data(self.metadata)
        self.logger.error(
            msg, extra={"json_fields": {**truncated_metadata, **truncated_extra}}
        )

    def debug(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        truncated_extra = self._truncate_large_data(extra)
        truncated_metadata = self._truncate_large_data(self.metadata)
        self.logger.debug(
            msg, extra={"json_fields": {**truncated_metadata, **truncated_extra}}
        )

    def exception(self, msg: str, **extra):
        msg = self._wrap(msg, **extra)
        truncated_extra = self._truncate_large_data(extra)
        truncated_metadata = self._truncate_large_data(self.metadata)
        self.logger.exception(
            msg, extra={"json_fields": {**truncated_metadata, **truncated_extra}}
        )

    def _wrap(self, msg: str, **extra):
        extra_msg = str(extra or "")
        text = f"{self.prefix} {msg} {extra_msg}"
        if len(text) > self.max_length:
            text = text[: self.max_length] + "..."
        return text

    def _truncate_large_data(self, data, max_size=10000):
        if isinstance(data, dict):
            return {k: self._truncate_large_data(v, max_size) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._truncate_large_data(v, max_size) for v in data[:100]]
        elif isinstance(data, str) and len(data) > max_size:
            return data[:max_size] + "... [truncated]"
        return data


class PrefixFilter(logging.Filter):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def filter(self, record):
        record.msg = f"{self.prefix} {record.msg}"
        return True
