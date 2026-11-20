import time

import pytest

from backend.blocks.code_extraction_block import CodeExtractionBlock


@pytest.mark.parametrize(
    "text, code",
    [
        ("```py\na = 1\n```", "a = 1"),
        ("```PY \t\n  a = 1  \n```\ntext\n```py\nb = 2\n```", "a = 1\n\nb = 2"),
        ("```py\na = 1\n```\n```py\nb = 2", "a = 1"),
        ("```py\na = 1\n```py\nb = 2\n```", "a = 1"),
        ("```py\n```js\na = 1\n```", "```js\na = 1"),
        ("```py\na = 1", ""),
        ("```py\n```", ""),
        ("```py\n\n```", ""),
        ("```python\na = 1\n```", ""),
    ],
)
def test_extract_code_takes_each_closed_fence_of_the_language(text: str, code: str):
    assert CodeExtractionBlock().extract_code(text, "py") == code


def test_extract_code_scans_unclosed_fences_in_linear_time():
    text = "```py\nx" * 20_000
    started = time.thread_time()
    assert CodeExtractionBlock().extract_code(text, "py") == ""
    # A pattern that rescans from every fence takes over 10 s on this input.
    assert time.thread_time() - started < 1.0


@pytest.mark.asyncio
async def test_run_preserves_source_order_across_aliases() -> None:
    block = CodeExtractionBlock()
    data = block.Input(
        text=(
            "Intro\n"
            "```py\nanswer = 42\n```\n"
            "```python\nprint(answer)\n```\n"
            "```pY\ndoubled = answer * 2\n```\n"
            "Outro"
        )
    )

    assert [result async for result in block.run(data)] == [
        ("python", "answer = 42\n\nprint(answer)\n\ndoubled = answer * 2"),
        ("remaining_text", "Intro\nOutro"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [
        ("```py\n\n```\n```python\n \t\n```", []),
        (
            "```py\n```\n```python\nanswer = 42\n```",
            [("python", "answer = 42")],
        ),
        (
            "```python\nanswer = 42\n```\n```py\n``` \t\n"
            "```python\nprint(answer)\n```",
            [("python", "answer = 42\n\nprint(answer)")],
        ),
        (
            "```py\nanswer = 42\n```\n```python\n\n```\n" "```pY\nprint(answer)\n```",
            [("python", "answer = 42\n\nprint(answer)")],
        ),
    ],
)
async def test_run_skips_blank_fences_across_aliases(
    text: str, expected: list[tuple[str, str]]
) -> None:
    block = CodeExtractionBlock()
    assert [result async for result in block.run(block.Input(text=text))] == expected
