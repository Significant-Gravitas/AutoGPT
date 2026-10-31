import pytest

from backend.blocks.code_extraction_block import CodeExtractionBlock


@pytest.mark.parametrize(
    "text, code",
    [
        ("```py\na = 1\n```", "a = 1"),
        ("```PY \t\n  a = 1  \n```\ntext\n```py\nb = 2\n```", "a = 1\n\nb = 2"),
        ("```py\na = 1\n```\n```py\nb = 2", "a = 1"),
        ("```py\na = 1\n```py\nb = 2\n```", "a = 1"),
        ("```py\na = 1", ""),
        ("```py\n```", ""),
        ("```py\n\n```", ""),
        ("```python\na = 1\n```", ""),
    ],
)
def test_extract_code_takes_each_closed_fence_of_the_language(text: str, code: str):
    assert CodeExtractionBlock().extract_code(text, "py") == code
