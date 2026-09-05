"""Logging safeguards loaded automatically by appliance Python processes."""

import logging


class _RedactWebSocketQuery(logging.Filter):
    """Strip bearer query strings from Uvicorn WebSocket status messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.name == "uvicorn.error"
            and isinstance(record.msg, str)
            and "WebSocket %s" in record.msg
            and isinstance(record.args, tuple)
            and len(record.args) >= 2
            and isinstance(record.args[1], str)
        ):
            args = list(record.args)
            args[1] = args[1].partition("?")[0]
            record.args = tuple(args)
        return True


# Uvicorn's HTTP access records include the complete target, including query
# parameters. Application logs remain enabled; only automatic request logs are
# suppressed. WebSocket lifecycle messages use uvicorn.error, so retain those
# after redacting their query string.
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.error").addFilter(_RedactWebSocketQuery())
