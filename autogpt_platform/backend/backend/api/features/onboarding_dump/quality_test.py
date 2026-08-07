"""Tests for the brain-dump quality gate.

The layering is the contract under test: unmistakable garbage and clearly
substantial dumps must be decided without a model call, only the ambiguous
middle may pay for one, and every judge failure resolves to a recoverable
rejection rather than a silent pass.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import quality

SUBSTANTIAL_DUMP = (
    "So I run a small bakery in Lisbon with my sister. Every Friday we get "
    "about forty wholesale orders by email and I retype them into a "
    "spreadsheet, then chase the suppliers on WhatsApp, then invoice "
    "everyone through our accounting tool at the end of the month. Honestly "
    "the invoicing and the order retyping are the parts I would love to "
    "never touch again."
)

SPANISH_DUMP = (
    "Trabajo como gestora de redes sociales para tres restaurantes en "
    "Madrid. Cada semana programo las publicaciones, respondo los "
    "comentarios, preparo un informe con las métricas de cada local y se lo "
    "envío a los dueños por correo. Me encantaría automatizar los informes "
    "y las respuestas más repetitivas para dedicar ese tiempo a las "
    "campañas nuevas."
)


def _llm_client(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.fixture
def llm(mocker: MockerFixture):
    def install(client: MagicMock | None) -> MagicMock:
        return mocker.patch.object(
            quality, "get_openai_client", MagicMock(return_value=client)
        )

    return install


# --- deterministic rejects: no model call ---------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcript",
    [
        "",
        "   \n\t  ",
        "... !!! ,,, --- ???",
        ". . . . . . . . . .",
    ],
)
async def test_empty_and_symbol_only_transcripts_reject_without_llm(
    llm, transcript: str
):
    client_factory = llm(_llm_client('{"usable": true}'))
    assert (
        await quality.check_transcript_quality(transcript) == quality.NO_USABLE_SPEECH
    )
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_repeated_word_loop_rejects_without_llm(llm):
    client_factory = llm(_llm_client('{"usable": true}'))
    looped = "you " * 30
    assert await quality.check_transcript_quality(looped) == quality.NO_USABLE_SPEECH
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_repeated_phrase_loop_rejects_without_llm(llm):
    client_factory = llm(_llm_client('{"usable": true}'))
    looped = "thank you for watching " * 12
    assert await quality.check_transcript_quality(looped) == quality.NO_USABLE_SPEECH
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_compressible_character_garbage_rejects_without_llm(llm):
    # One "word", so the unique-ratio check can't see it — the zlib
    # compression check is what catches a decoder stuck on a character.
    client_factory = llm(_llm_client('{"usable": true}'))
    garbage = "a" * 200
    assert await quality.check_transcript_quality(garbage) == quality.NO_USABLE_SPEECH
    client_factory.assert_not_called()


# --- deterministic passes: no model call ----------------------------------


@pytest.mark.asyncio
async def test_substantial_english_dump_passes_without_llm(llm):
    client_factory = llm(_llm_client('{"usable": false}'))
    assert await quality.check_transcript_quality(SUBSTANTIAL_DUMP) is None
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_substantial_non_english_dump_passes_without_llm(llm):
    client_factory = llm(_llm_client('{"usable": false}'))
    assert await quality.check_transcript_quality(SPANISH_DUMP) is None
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_spaceless_script_dump_passes_on_characters(llm):
    # split() sees a handful of "words" here; the character count is what
    # recognizes a substantial CJK dump.
    client_factory = llm(_llm_client('{"usable": false}'))
    chinese = (
        "我在上海经营一家小型设计工作室，主要为餐饮品牌做菜单和门店视觉设计。"
        "每周我要整理客户反馈、更新项目进度表、给供应商发对账单，还要在社交媒体"
        "上发布作品集。我最想把整理反馈和发对账单这两件事自动化，因为它们每周"
        "都要花掉我好几个小时。"
    )
    assert await quality.check_transcript_quality(chinese) is None
    client_factory.assert_not_called()


# --- the ambiguous middle: one LLM call decides ---------------------------


@pytest.mark.asyncio
async def test_short_meaningful_request_passes_via_llm(llm):
    llm(_llm_client('{"usable": true}'))
    short = "Automate my Shopify refund emails."
    assert await quality.check_transcript_quality(short) is None


@pytest.mark.asyncio
async def test_short_filler_rejects_via_llm(llm):
    llm(_llm_client('{"usable": false}'))
    filler = "Uh hello hello, testing, can you hear me?"
    assert (
        await quality.check_transcript_quality(filler) == quality.INSUFFICIENT_CONTENT
    )


@pytest.mark.asyncio
async def test_hallucinated_outro_rejects_via_llm(llm):
    llm(_llm_client('{"usable": false}'))
    assert (
        await quality.check_transcript_quality("Thank you for watching.")
        == quality.INSUFFICIENT_CONTENT
    )


@pytest.mark.asyncio
async def test_llm_verdict_tolerates_markdown_fence(llm):
    llm(_llm_client('```json\n{"usable": true}\n```'))
    assert await quality.check_transcript_quality("Automate my invoices.") is None


# --- judge failures: never a silent pass ----------------------------------


@pytest.mark.asyncio
async def test_no_llm_client_rejects_ambiguous_input(llm):
    llm(None)
    assert (
        await quality.check_transcript_quality("Automate my invoices.")
        == quality.INSUFFICIENT_CONTENT
    )


@pytest.mark.asyncio
async def test_no_llm_client_still_passes_clear_input(llm):
    llm(None)
    assert await quality.check_transcript_quality(SUBSTANTIAL_DUMP) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json at all", '{"usable": "yes"}', "{}"])
async def test_malformed_llm_verdict_rejects(llm, content: str):
    llm(_llm_client(content))
    assert (
        await quality.check_transcript_quality("Automate my invoices.")
        == quality.INSUFFICIENT_CONTENT
    )


@pytest.mark.asyncio
async def test_empty_choices_rejects(llm):
    response = MagicMock()
    response.choices = []
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    llm(client)
    assert (
        await quality.check_transcript_quality("Automate my invoices.")
        == quality.INSUFFICIENT_CONTENT
    )


@pytest.mark.asyncio
async def test_llm_timeout_rejects(llm):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError())
    llm(client)
    assert (
        await quality.check_transcript_quality("Automate my invoices.")
        == quality.INSUFFICIENT_CONTENT
    )


@pytest.mark.asyncio
async def test_llm_error_rejects(llm):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    llm(client)
    assert (
        await quality.check_transcript_quality("Automate my invoices.")
        == quality.INSUFFICIENT_CONTENT
    )
