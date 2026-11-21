"""TranscribeSpeechBlock against a stubbed Scribe HTTP response: what it sends,
how it splits the words by speaker, and what a platform-key run is charged."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.blocks.elevenlabs._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.elevenlabs.speech_to_text import (
    SCRIBE_MODEL,
    SCRIBE_URL,
    SpeechSegment,
    TranscribeSpeechBlock,
)
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.billing import charge_reconciled_usage
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import elevenlabs_credentials

PLATFORM_KEY_INPUT = {
    "id": elevenlabs_credentials.id,
    "provider": elevenlabs_credentials.provider,
    "type": elevenlabs_credentials.type,
    "title": elevenlabs_credentials.title,
}


def _word(text: str, start: float, end: float, speaker: str, kind: str = "word"):
    return {
        "text": text,
        "type": kind,
        "start": start,
        "end": end,
        "speaker_id": speaker,
        "logprob": -0.1,
    }


A_THEN_B_THEN_A = {
    "language_code": "eng",
    "language_probability": 0.99,
    "text": "Shall we start? Yes. Good.",
    "audio_duration_secs": 4.2,
    "words": [
        _word("Shall", 0.1, 0.4, "speaker_0"),
        _word(" ", 0.4, 0.5, "speaker_0", "spacing"),
        _word("we", 0.5, 0.6, "speaker_0"),
        _word(" ", 0.6, 0.7, "speaker_0", "spacing"),
        _word("start?", 0.7, 1.1, "speaker_0"),
        _word(" ", 1.1, 1.8, "speaker_0", "spacing"),
        _word("Yes.", 1.8, 2.2, "speaker_1"),
        _word(" ", 2.2, 3.0, "speaker_1", "spacing"),
        _word("Good.", 3.0, 3.4, "speaker_0"),
    ],
}


@pytest.mark.parametrize(
    "options, sent_options",
    [
        ({}, {}),
        (
            {"language": "nl", "max_speakers": 3},
            {"language_code": "nl", "num_speakers": "3"},
        ),
    ],
    ids=["defaults", "language-and-speaker-limit"],
)
async def test_a_recording_comes_back_split_by_speaker(options, sent_options):
    outputs, post = await _run(
        A_THEN_B_THEN_A, {"credentials": PLATFORM_KEY_INPUT, **options}
    )

    assert post.await_args.args == (SCRIBE_URL,)
    sent = post.await_args.kwargs
    assert sent["headers"] == {
        "xi-api-key": TEST_CREDENTIALS.api_key.get_secret_value()
    }
    assert sent["data"] == {"model_id": SCRIBE_MODEL, "diarize": "true", **sent_options}
    assert sent["files"][0][1][0] == "meeting.mp3"

    assert outputs["transcript"] == "Shall we start? Yes. Good."
    assert outputs["segments"] == [
        SpeechSegment(speaker="speaker_0", start=0.1, end=1.1, text="Shall we start?"),
        SpeechSegment(speaker="speaker_1", start=1.8, end=2.2, text="Yes."),
        SpeechSegment(speaker="speaker_0", start=3.0, end=3.4, text="Good."),
    ]
    assert outputs["speakers"] == ["speaker_0", "speaker_1"]


@pytest.mark.parametrize(
    "reported_seconds, last_word_end, credits",
    [
        (3600.0, 60.0, 30),  # $0.20 an hour at 150 cr/$, from the reported length
        (None, 3600.0, 30),  # no duration reported: bill up to the last word
        (4.2, 3.4, 1),  # a short clip still costs the 1-credit minimum
    ],
    ids=["one-hour", "duration-missing", "short-clip"],
)
async def test_a_platform_key_run_is_charged_for_the_audio_scribe_reports(
    reported_seconds, last_word_end, credits
):
    payload = {
        **A_THEN_B_THEN_A,
        "audio_duration_secs": reported_seconds,
        "words": [_word("Hi.", last_word_end - 0.5, last_word_end, "speaker_0")],
    }
    inputs = {"credentials": PLATFORM_KEY_INPUT}
    block = TranscribeSpeechBlock()
    await _run(payload, inputs, block)

    db_client = MagicMock()
    db_client.spend_credits = AsyncMock(return_value=500)
    with (
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=db_client,
        ),
        patch("backend.executor.billing.get_db_client", return_value=MagicMock()),
        patch("backend.executor.billing.handle_low_balance"),
    ):
        delta, _ = await charge_reconciled_usage(
            _node_exec(block.id, inputs), block.execution_stats, pre_flight_charge=0
        )

    assert delta == credits
    assert db_client.spend_credits.await_args.kwargs["cost"] == credits


def test_a_run_on_the_users_own_key_is_not_charged():
    cost, _ = block_usage_cost(
        block=TranscribeSpeechBlock(),
        input_data={"credentials": TEST_CREDENTIALS_INPUT},
        stats=NodeExecutionStats(provider_cost=0.2, provider_cost_type="cost_usd"),
    )
    assert cost == 0


async def _run(
    payload: dict, inputs: dict, block: TranscribeSpeechBlock | None = None
) -> tuple[dict, AsyncMock]:
    block = block or TranscribeSpeechBlock()
    response = MagicMock()
    response.json.return_value = payload
    post = AsyncMock(return_value=response)
    with (
        patch("backend.blocks.elevenlabs.speech_to_text.Requests.post", post),
        patch.object(
            TranscribeSpeechBlock,
            "read_audio",
            AsyncMock(return_value=("meeting.mp3", b"ID3")),
        ),
    ):
        outputs = {
            name: value
            async for name, value in block.run(
                TranscribeSpeechBlock.Input.model_validate(
                    {"audio": "workspace://meeting", **inputs}
                ),
                credentials=TEST_CREDENTIALS,
                execution_context=ExecutionContext(user_id="user-a"),
            )
        }
    return outputs, post


def _node_exec(block_id: str, inputs: dict) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-a",
        graph_exec_id=str(uuid4()),
        graph_id=str(uuid4()),
        graph_version=1,
        node_exec_id=str(uuid4()),
        node_id=str(uuid4()),
        block_id=block_id,
        inputs=inputs,
        execution_context=ExecutionContext(user_id="user-a"),
    )
