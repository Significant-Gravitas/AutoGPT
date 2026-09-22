"""Unit tests for Meeting BaaS duration-based cost emission."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.baas.bots import (
    _MEETING_BAAS_USD_PER_SECOND,
    BaasBotFetchMeetingDataBlock,
)
from backend.data.model import APIKeyCredentials, NodeExecutionStats

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="baas",
    title="Mock BaaS API Key",
    api_key=SecretStr("mock-baas-api-key"),
    expires_at=None,
)


def test_usd_per_second_derives_from_published_rate():
    """$0.69/hour published rate → ~$0.000192/second."""
    assert _MEETING_BAAS_USD_PER_SECOND == pytest.approx(0.69 / 3600)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "duration_seconds, expected_usd",
    [
        (3600, 0.69),  # 1 hour
        (1800, 0.345),  # 30 min
        (0, None),  # no recording → no emission
        (None, None),  # missing duration field → no emission
    ],
)
async def test_fetch_meeting_data_emits_duration_cost_usd(
    duration_seconds, expected_usd
):
    """FetchMeetingData extracts duration_seconds from bot metadata and
    emits provider_cost / cost_usd scaled by the published $0.69/hr rate.
    Emission is skipped when duration is 0 or missing.
    """
    block = BaasBotFetchMeetingDataBlock()

    bot_meta = {"id": "bot-xyz"}
    if duration_seconds is not None:
        bot_meta["duration_seconds"] = duration_seconds

    mock_api = AsyncMock()
    mock_api.get_meeting_data.return_value = {
        "mp4": "https://example/recording.mp4",
        "bot_data": {"bot": bot_meta, "transcripts": []},
    }

    captured: list[NodeExecutionStats] = []
    with (
        patch("backend.blocks.baas.bots.MeetingBaasAPI", return_value=mock_api),
        patch.object(block, "merge_stats", side_effect=captured.append),
    ):
        outputs = []
        async for name, val in block.run(
            block.input_schema(
                credentials={
                    "id": TEST_CREDENTIALS.id,
                    "provider": TEST_CREDENTIALS.provider,
                    "type": TEST_CREDENTIALS.type,
                },
                bot_id="bot-xyz",
                include_transcripts=False,
            ),
            credentials=TEST_CREDENTIALS,
        ):
            outputs.append((name, val))

    # Always yields the 3 outputs regardless of duration.
    names = [n for n, _ in outputs]
    assert "mp4_url" in names and "metadata" in names

    if expected_usd is None:
        assert captured == []
    else:
        assert len(captured) == 1
        assert captured[0].provider_cost == pytest.approx(expected_usd)
        assert captured[0].provider_cost_type == "cost_usd"
