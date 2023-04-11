from os import environ


def env(key, type_, default=None):
    if key not in environ:
        return default

    val = environ[key]

    if type_ == str:
        return val
    elif type_ == bool:
        if val.lower() in ["1", "true", "yes", "y", "ok", "on"]:
            return True
        if val.lower() in ["0", "false", "no", "n", "nok", "off"]:
            return False
        raise ValueError(
            "Invalid environment variable '%s' (expected a boolean): '%s'" % (key, val)
        )
    elif type_ == int:
        try:
            return int(val)
        except ValueError:
            raise ValueError(
                "Invalid environment variable '%s' (expected an integer): '%s'" % (key, val)
            ) from None


LOGURU_AUTOINIT = env("LOGURU_AUTOINIT", bool, True)

LOGURU_FORMAT = env(
    "LOGURU_FORMAT",
    str,
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
LOGURU_FILTER = env("LOGURU_FILTER", str, None)
LOGURU_LEVEL = env("LOGURU_LEVEL", str, "DEBUG")
LOGURU_COLORIZE = env("LOGURU_COLORIZE", bool, None)
LOGURU_SERIALIZE = env("LOGURU_SERIALIZE", bool, False)
LOGURU_BACKTRACE = env("LOGURU_BACKTRACE", bool, True)
LOGURU_DIAGNOSE = env("LOGURU_DIAGNOSE", bool, True)
LOGURU_ENQUEUE = env("LOGURU_ENQUEUE", bool, False)
LOGURU_CATCH = env("LOGURU_CATCH", bool, True)

LOGURU_TRACE_NO = env("LOGURU_TRACE_NO", int, 5)
LOGURU_TRACE_COLOR = env("LOGURU_TRACE_COLOR", str, "<cyan><bold>")
LOGURU_TRACE_ICON = env("LOGURU_TRACE_ICON", str, "\u270F\uFE0F")  # Pencil

LOGURU_DEBUG_NO = env("LOGURU_DEBUG_NO", int, 10)
LOGURU_DEBUG_COLOR = env("LOGURU_DEBUG_COLOR", str, "<blue><bold>")
LOGURU_DEBUG_ICON = env("LOGURU_DEBUG_ICON", str, "\U0001F41E")  # Lady Beetle

LOGURU_INFO_NO = env("LOGURU_INFO_NO", int, 20)
LOGURU_INFO_COLOR = env("LOGURU_INFO_COLOR", str, "<bold>")
LOGURU_INFO_ICON = env("LOGURU_INFO_ICON", str, "\u2139\uFE0F")  # Information

LOGURU_SUCCESS_NO = env("LOGURU_SUCCESS_NO", int, 25)
LOGURU_SUCCESS_COLOR = env("LOGURU_SUCCESS_COLOR", str, "<green><bold>")
LOGURU_SUCCESS_ICON = env("LOGURU_SUCCESS_ICON", str, "\u2705")  # White Heavy Check Mark

LOGURU_WARNING_NO = env("LOGURU_WARNING_NO", int, 30)
LOGURU_WARNING_COLOR = env("LOGURU_WARNING_COLOR", str, "<yellow><bold>")
LOGURU_WARNING_ICON = env("LOGURU_WARNING_ICON", str, "\u26A0\uFE0F")  # Warning

LOGURU_ERROR_NO = env("LOGURU_ERROR_NO", int, 40)
LOGURU_ERROR_COLOR = env("LOGURU_ERROR_COLOR", str, "<red><bold>")
LOGURU_ERROR_ICON = env("LOGURU_ERROR_ICON", str, "\u274C")  # Cross Mark

LOGURU_CRITICAL_NO = env("LOGURU_CRITICAL_NO", int, 50)
LOGURU_CRITICAL_COLOR = env("LOGURU_CRITICAL_COLOR", str, "<RED><bold>")
LOGURU_CRITICAL_ICON = env("LOGURU_CRITICAL_ICON", str, "\u2620\uFE0F")  # Skull and Crossbones
