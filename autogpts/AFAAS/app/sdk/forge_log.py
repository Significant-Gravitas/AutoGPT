import json
import logging
import logging.config
import logging.handlers
import os
import queue

JSON_LOGGING = os.environ.get("JSON_LOGGING", "false").lower() == "true"

CHAT = 29
NOTICE = 15
TRACE = 5
logging.addLevelName(CHAT, "CHAT")
logging.addLevelName(NOTICE, "NOTICE")
logging.addLevelName(TRACE, "TRACE")

RESET_SEQ: str = "\033[0m"
COLOR_SEQ: str = "\033[1;%dm"
BOLD_SEQ: str = "\033[1m"
UNDERLINE_SEQ: str = "\033[04m"

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
        if self.use_color and levelname in KEYWORD_COLORS:
            levelname_color = KEYWORD_COLORS[levelname] + levelname + RESET_SEQ
            rec.levelname = levelname_color
        rec.name = f"{GREY}{os.path.relpath(rec.pathname):<15}{RESET_SEQ}"
        rec.msg = (
            KEYWORD_COLORS[levelname] + EMOJIS[levelname] + "  " + str( rec.msg )+ RESET_SEQ
        )

        message = logging.Formatter.format(self, rec)
        if rec.levelno == logging.DEBUG  and len(message) > 1000:
            message = message[:800] + '[...] ' + os.path.abspath(ForgeLogger.LOG_FILENAME)
        return message



class ForgeLogger(logging.Logger):
    """
    This adds extra logging functions such as logger.trade and also
    sets the logger to use the custom formatter
    """

    LOG_FILENAME = 'debug.log' 
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    CONSOLE_FORMAT: str = (
        "[%(asctime)s] [$BOLD%(name)-15s,%(lineno)d$RESET] [%(levelname)-8s]\t%(message)s"
    )
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


    def __init__(self, name: str, logLevel: str = "DEBUG", log_folder: str = './'):
        if hasattr(self, '_initialized'):
            return
        
        logging.Logger.__init__(self, name, logLevel)
        # self.log_folder = log_folder

        # Queue Handler
        queue_handler = logging.handlers.QueueHandler(queue.Queue(-1))
        json_formatter = logging.Formatter(self.JSON_FORMAT)
        queue_handler.setFormatter(json_formatter)
        self.addHandler(queue_handler)

        # File Handler
        os.makedirs(log_folder, exist_ok=True)
        log_file_path = os.path.join(log_folder, self.LOG_FILENAME)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file_path, when="midnight", interval=1, backupCount=7
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = r"^\d{4}-\d{2}-\d{2}.log$"
        file_handler.setLevel(logging.DEBUG) 
        file_handler.setFormatter(logging.Formatter(self.FORMAT))  # Use a simple format for file logs
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
        
    def notice(self, msg, *args, **kwargs):
      
        if self.isEnabledFor(NOTICE):
            self._log(NOTICE, msg, args, **kwargs)


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
            "format": ForgeLogger.COLOR_FORMAT,
        },
    },
    handlers={
        "h": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": logging.INFO,
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": ForgeLogger.LOG_FILENAME,
            "formatter": "console",
            "level": TRACE,
        },
    },
    root={
        "handlers": ["h", "file"],
        "level": logging.INFO,
    },
    loggers={
        "autogpt": {
            "handlers": ["h", "file"],
            "level": logging.INFO,
            "propagate": False,
        },
    },
)


def setup_logger():
    """
    Setup the logger with the specified format
    """
    logging.config.dictConfig(logging_config)

def get_client_logger(logger_level: int = logging.DEBUG):
    # Configure logging before we do anything else.
    # Application logs need a place to live.
    client_logger = logging.getLogger("autogpt_client_application")
    client_logger.setLevel(logger_level)

    formatter = logging.Formatter(
        "%(asctime)s#%(filename)s:%(funcName)s:%(levelname)s:%(message)s"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logger_level)
    ch.setFormatter(formatter)

    client_logger.addHandler(ch)

    return client_logger