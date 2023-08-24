import logging

from colorama import Fore, Style

from .utils import remove_color_codes


class AutoGptFormatter(logging.Formatter):
    """
    Allows to handle custom placeholders 'title_color' and 'message_no_color'.
    To use this formatter, make sure to pass 'color', 'title' as log extras.
    """

    # level -> (level & text color, title color)
    LEVEL_COLOR_MAP = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Make sure `msg` is a string
        if not hasattr(record, "msg"):
            record.msg = ""
        elif not type(record.msg) == str:
            record.msg = str(record.msg)

        # Strip color from the message to prevent color spoofing
        if record.msg and not getattr(record, "preserve_color", False):
            record.msg = remove_color_codes(record.msg)

        # Determine default color based on error level
        level_color = ""
        if record.levelno in self.LEVEL_COLOR_MAP:
            level_color = self.LEVEL_COLOR_MAP[record.levelno]
            record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"

        # Determine color for message
        color = getattr(record, "color", level_color)
        color_is_specified = hasattr(record, "color")

        # Determine color for title
        title = getattr(record, "title", "")
        title_color = getattr(record, "title_color", "") or level_color
        if title and title_color:
            title = f"{title_color + Style.BRIGHT}{title}{Style.RESET_ALL}"
        # Make sure record.title is set, and padded with a space if not empty
        record.title = f"{title} " if title else ""

        # Don't color INFO messages unless the color is explicitly specified.
        if color and (record.levelno != logging.INFO or color_is_specified):
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"

        return super().format(record)
