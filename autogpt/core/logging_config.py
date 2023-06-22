import logging

logging_config = dict(
    version=1,
    formatters={
        "file_format": {
            "format": "%(levelname)-8s %(asctime)s %(name)-50s %(message)s"
        },
        "console": {"format": "%(levelname)-8s %(asctime)s %(name)-50s %(message)s"},
    },
    handlers={
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "file_format",
            "maxBytes": 10485760,
            "backupCount": 10,
            "level": logging.INFO,
            "filename": "auto-gpt.log",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "file_format",
            "maxBytes": 10485760,
            "backupCount": 10,
            "level": logging.ERROR,
            "filename": "auto-gpt.errors",
        },
        "h": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": logging.INFO,
        },
    },
    root={
        "handlers": ["file", "error_file", "h"],
        "level": logging.DEBUG,
    },
)
