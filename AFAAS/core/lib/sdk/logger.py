import json
import logging
import logging.config
import logging.handlers
import os
import queue

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

CONSOLE_LOG_LEVEL = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", "DEBUG").upper()

JSON_LOGGING = os.environ.get("JSON_LOGGING", "false").lower() == "true"

CHAT = 29
DB_LOG = 18
NOTICE = 15
TRACE = 5
logging.addLevelName(CHAT, "CHAT")
logging.addLevelName(NOTICE, "NOTICE")
logging.addLevelName(DB_LOG, "DB_LOG")
logging.addLevelName(TRACE, "TRACE")

RESET_SEQ: str = "\033[0m"
COLOR_SEQ: str = "\033[1;%dm"
BOLD_SEQ: str = "\033[1m"
UNDERLINE_SEQ: str = "\033[04m"
ITALIC_SEQ = "\033[3m"

ORANGE: str = "\033[33m"
YELLOW: str = "\033[93m"
WHITE: str = "\33[37m"
BLUE: str = "\033[34m"
LIGHT_BLUE: str = "\033[94m"
RED: str = "\033[91m"
GREY: str = "\33[90m"
GREEN: str = "\033[92m"
PURPLE: str = "\033[35m"
BRIGHT_PINK: str = "\033[95m"


EMOJIS: dict[str, str] = {
    "TRACE": "🔍",
    "DEBUG": "🐛",
    "INFO": "📝",
    "CHAT": "💬",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "💥",
    "NOTICE": "🔊",
    "DB_LOG": "📁",
}

KEYWORD_COLORS: dict[str, str] = {
    "TRACE": BRIGHT_PINK,
    "DEBUG": WHITE,
    "INFO": LIGHT_BLUE,
    "CHAT": PURPLE,
    "NOTICE": GREEN,
    "WARNING": YELLOW,
    "ERROR": ORANGE,
    "CRITICAL": RED,
    "DB_LOG": GREY,
}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(record.__dict__)


def formatter_message(message: str, use_color: bool = True) -> str:
    """
    Syntax highlight certain keywords
    """
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


def format_word(
    message: str, word: str, color_seq: str, bold: bool = False, underline: bool = False
) -> str:
    """
    Surround the fiven word with a sequence
    """
    replacer = color_seq + word + RESET_SEQ
    if underline:
        replacer = UNDERLINE_SEQ + replacer
    if bold:
        replacer = BOLD_SEQ + replacer
    return message.replace(word, replacer)


class ConsoleFormatter(logging.Formatter):
    """
    This Formatted simply colors in the levelname i.e 'INFO', 'DEBUG'
    """

    def __init__(
        self, fmt: str, datefmt: str = None, style: str = "%", use_color: bool = True
    ):
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color
        self.default_msec_format = None

    def format(self, record: logging.LogRecord) -> str:
        """
        Format and highlight certain keywords
        """
        rec = record
        levelname = rec.levelname
        current_color = ''

        if self.use_color and levelname in KEYWORD_COLORS:
            current_color = KEYWORD_COLORS[levelname]
            levelname_color = current_color + levelname + RESET_SEQ
            rec.levelname = levelname_color

        rec.name = f"{GREY}{os.path.relpath(rec.pathname):<15}{RESET_SEQ}"
        rec.msg = (
            current_color
            + EMOJIS[levelname]
            + "  "
            + str(rec.msg)
            + RESET_SEQ
        )

        message = logging.Formatter.format(self, rec)

        # Reinstate color after each reset
        if self.use_color:
            message = message.replace(RESET_SEQ, RESET_SEQ + current_color)

        #if rec.levelno == logging.DEBUG and len(message) > 1000:
        if rec.levelno == TRACE and len(message) > 1000:
            message = (
                message[:800] + "[...] " + os.path.abspath(AFAASLogger.LOG_FILENAME)
            )
        return message


class AFAASLogger(logging.Logger):
    """
    This adds extra logging functions such as LOG.trade and also
    sets the logger to use the custom formatter
    """

    LOG_FILENAME = "debug.log"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    CONSOLE_FORMAT: str = "[%(asctime)s] [$BOLD%(name)-15s,%(lineno)d$RESET] [%(levelname)-8s]\t%(message)s"
    FORMAT: str = "%(asctime)s %(name)-15s %(levelname)-8s %(message)s"
    COLOR_FORMAT: str = formatter_message(CONSOLE_FORMAT, True)
    JSON_FORMAT: str = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

    # _instance = None  # Class attribute to hold the single instance

    # @classmethod
    # def get_instance(cls, name: str, logLevel: str = "DEBUG", log_folder: str = './'):
    #     if cls._instance is None:
    #         cls._instance = cls(name, logLevel, log_folder)

    #     cls._instance.name = name
    #     cls._instance.level = logLevel
    #     return cls._instance

    def __init__(self, name: str, log_folder: str = "./"):
        if hasattr(self, "_initialized"):
            return

        logging.Logger.__init__(self, name)
        # self.log_folder = log_folder

        # Queue Handler
        queue_handler = logging.handlers.QueueHandler(queue.Queue(-1))
        json_formatter = logging.Formatter(self.JSON_FORMAT)
        queue_handler.setFormatter(json_formatter)
        self.addHandler(queue_handler)

        # File Handler
        os.makedirs(log_folder, exist_ok=True)
        log_file_path = os.path.join(log_folder, LOG_FILENAME)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file_path, when="midnight", interval=1, backupCount=7
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = r"^\d{4}-\d{2}-\d{2}.log$"
        file_handler.setFormatter(
            logging.Formatter(self.FORMAT)
        )  # Use a simple format for file logs
        self.addHandler(file_handler)

        if JSON_LOGGING:
            console_formatter = JsonFormatter()
        else:
            console_formatter = ConsoleFormatter(self.COLOR_FORMAT)
        console = logging.StreamHandler()
        console.setFormatter(console_formatter)
        self.addHandler(console)

        # self._initialized = True

    def chat(self, role: str, openai_repsonse: dict, messages=None, *args, **kws):
        """
        Parse the content, log the message and extract the usage into prometheus metrics
        """
        role_emojis = {
            "system": "🖥️",
            "user": "👤",
            "assistant": "🤖",
            "function": "⚙️",
        }
        if self.isEnabledFor(CHAT):
            if messages:
                for message in messages:
                    self._log(
                        CHAT,
                        f"{role_emojis.get(message['role'], '🔵')}: {message['content']}",
                    )
            else:
                response = json.loads(openai_repsonse)

                self._log(
                    CHAT,
                    f"{role_emojis.get(role, '🔵')}: {response['choices'][0]['message']['content']}",
                )

    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, kwargs, stacklevel=2)

    def notice(self, msg, *args, **kwargs):
        if self.isEnabledFor(NOTICE):
            self._log(NOTICE, msg, args, kwargs, stacklevel=2)

    def db_log(self, msg, *args, **kwargs):
        if self.isEnabledFor(DB_LOG):
            self._log(DB_LOG, msg, args, kwargs, stacklevel=2)


    @staticmethod
    def bold(msg: str) -> str:
        """
        Returns the message in bold
        """
        return BOLD_SEQ + msg + RESET_SEQ

    @staticmethod
    def italic(msg: str) -> str:
        """
        Returns the message in italic
        """
        return ITALIC_SEQ + msg + RESET_SEQ


class QueueLogger(logging.Logger):
    """
    Custom logger class with queue
    """

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        queue_handler = logging.handlers.QueueHandler(queue.Queue(-1))
        self.addHandler(queue_handler)


logging_config: dict = dict(
    version=1,
    formatters={
        "console": {
            "()": ConsoleFormatter,
            "format": AFAASLogger.COLOR_FORMAT,
        },
    },
    handlers={
        "h": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": CONSOLE_LOG_LEVEL,
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": AFAASLogger.LOG_FILENAME,
            "formatter": "console",
            "level": FILE_LOG_LEVEL,
        },
    },
    root={
        "handlers": ["h", "file"],
        "level": CONSOLE_LOG_LEVEL,
    },
    loggers={
        "autogpt": {
            "handlers": ["h", "file"],
            "level": CONSOLE_LOG_LEVEL,
            "propagate": False,
        },
    },
)


def setup_logger():
    """
    Setup the logger with the specified format
    """
    logging.config.dictConfig(logging_config)


LOG = AFAASLogger(name=__name__)
LOG.warning(f"Console log level is  : {logging.getLevelName(CONSOLE_LOG_LEVEL)}" )
LOG.warning(f"File log level is  : {logging.getLevelName(FILE_LOG_LEVEL)}" )
