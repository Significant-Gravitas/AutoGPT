import re


def remove_color_codes(s: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", s)


def fmt_kwargs(kwargs: dict) -> str:
    return ", ".join(f"{n}={repr(v)}" for n, v in kwargs.items())
