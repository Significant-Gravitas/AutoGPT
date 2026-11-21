import pytest

from backend.util.sql import escape_like


@pytest.mark.parametrize(
    "raw, escaped",
    [
        ("plain", "plain"),
        ("first_last", r"first\_last"),
        ("%%%", r"\%\%\%"),
        ("___", r"\_\_\_"),
        ("d%h", r"d\%h"),
        # Backslash is escaped first, so it never re-escapes the others.
        ("a\\_b", r"a\\\_b"),
        ("\\", r"\\"),
        # Other characters, including LIKE-irrelevant punctuation, pass through.
        ("it's [x] -- /* */", "it's [x] -- /* */"),
        ("夜🙂", "夜🙂"),
        ("", ""),
    ],
)
def test_escape_like(raw: str, escaped: str):
    assert escape_like(raw) == escaped


def test_escape_like_rejects_nul():
    with pytest.raises(ValueError, match="NUL"):
        escape_like("dan\x00a")
