import pytest

from backend.blocks.text import (
    MAX_TEXT_OUTPUT_LENGTH,
    CombineTextsBlock,
    TextReplaceBlock,
)

LIMIT_MESSAGE = f"over the limit of {MAX_TEXT_OUTPUT_LENGTH:,} characters"


class _UnbuiltStr(str):
    """A str whose output-building methods fail, so a test can prove the
    limit is checked before any output is allocated."""

    def replace(self, *args, **kwargs):
        raise AssertionError("output was built before the length check")

    def join(self, *args, **kwargs):
        raise AssertionError("output was built before the length check")


async def _outputs(block, input_data) -> list[tuple[str, object]]:
    return [(name, value) async for name, value in block.run(input_data)]


def _replace_input(text: str, old: str, new: str) -> TextReplaceBlock.Input:
    return TextReplaceBlock.Input.model_construct(text=text, old=old, new=new)


def _combine_input(texts: list[str], delimiter: str) -> CombineTextsBlock.Input:
    return CombineTextsBlock.Input.model_construct(input=texts, delimiter=delimiter)


@pytest.mark.asyncio
async def test_text_replace_refuses_empty_old_over_the_limit():
    block = TextReplaceBlock()
    input_data = _replace_input(
        _UnbuiltStr("a"), "", "b" * (MAX_TEXT_OUTPUT_LENGTH // 2)
    )
    with pytest.raises(ValueError, match=LIMIT_MESSAGE):
        await _outputs(block, input_data)


@pytest.mark.asyncio
async def test_text_replace_allows_large_new_when_old_is_absent():
    block = TextReplaceBlock()
    new = "b" * (MAX_TEXT_OUTPUT_LENGTH + 1)
    out = await _outputs(block, block.Input(text="hello", old="zzz", new=new))
    assert out == [("output", "hello")]


@pytest.mark.asyncio
async def test_text_replace_output_limit_boundary():
    block = TextReplaceBlock()
    at_limit = block.Input(text="ab", old="b", new="c" * (MAX_TEXT_OUTPUT_LENGTH - 1))
    [(name, value)] = await _outputs(block, at_limit)
    assert name == "output"
    assert len(value) == MAX_TEXT_OUTPUT_LENGTH

    over_limit = _replace_input(_UnbuiltStr("ab"), "b", "c" * MAX_TEXT_OUTPUT_LENGTH)
    with pytest.raises(ValueError, match=LIMIT_MESSAGE):
        await _outputs(block, over_limit)


@pytest.mark.asyncio
async def test_combine_texts_output_limit_boundary():
    block = CombineTextsBlock()
    at_limit = block.Input(
        input=["a", "b"], delimiter="d" * (MAX_TEXT_OUTPUT_LENGTH - 2)
    )
    [(name, value)] = await _outputs(block, at_limit)
    assert name == "output"
    assert len(value) == MAX_TEXT_OUTPUT_LENGTH

    over_limit = _combine_input(
        ["a", "b"], _UnbuiltStr("d" * (MAX_TEXT_OUTPUT_LENGTH - 1))
    )
    with pytest.raises(ValueError, match=LIMIT_MESSAGE):
        await _outputs(block, over_limit)
