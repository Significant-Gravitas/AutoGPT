"""Tests for the provider-recommendation job's parsing and read path.

The LLM call itself is not under test — what matters is that hallucinated
provider ids never reach storage, and that ``ready`` reflects the job (a
null column) rather than the greeting pipeline's status.
"""

from unittest.mock import AsyncMock

import pytest
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import recommend, service

USER_ID = "user-1"

KNOWN = {"slack", "github", "notion", "google"}


def test_parse_recommendations_drops_unknown_and_duplicate_providers():
    data = {
        "providers": [
            {"provider": "slack", "reason": "You mentioned Slack standups."},
            {"provider": "slack", "reason": "Duplicate."},
            {"provider": "made_up_tool", "reason": "Hallucinated."},
            {"provider": "github", "reason": ""},
            "not-a-dict",
            {"reason": "missing provider key"},
        ]
    }
    result = recommend._parse_recommendations(data, KNOWN)
    assert [r.provider for r in result] == ["slack", "github"]
    assert result[0].reason == "You mentioned Slack standups."


def test_parse_recommendations_caps_count_and_reason_length():
    known = {f"tool{i}" for i in range(10)}
    data = {
        "providers": [{"provider": f"tool{i}", "reason": "x" * 500} for i in range(10)]
    }
    result = recommend._parse_recommendations(data, known)
    assert len(result) == recommend.MAX_RECOMMENDATIONS
    assert all(len(r.reason) == recommend.MAX_REASON_CHARS for r in result)


def test_parse_recommendations_rejects_non_object_payloads():
    assert recommend._parse_recommendations(None, KNOWN) == []
    assert recommend._parse_recommendations(["slack"], KNOWN) == []
    assert recommend._parse_recommendations({"providers": "slack"}, KNOWN) == []


def _dump(**overrides) -> OnboardingBrainDump:
    fields = {
        "userId": USER_ID,
        "recordingId": "rec-1",
        "status": BrainDumpStatus.completed,
        "inputMode": BrainDumpInputMode.voice,
        "transcript": "I live in Slack and ship on GitHub.",
        "recommendedProviders": None,
        **overrides,
    }
    return OnboardingBrainDump.model_construct(**fields)


@pytest.mark.asyncio
async def test_recommended_providers_not_ready_while_column_is_null(
    mocker: MockerFixture,
):
    # Greeting pipeline already completed — status must not flip ready.
    mocker.patch.object(service.db, "get_dump", AsyncMock(return_value=_dump()))
    response = await service.get_recommended_providers(USER_ID)
    assert response.ready is False
    assert response.providers == []


@pytest.mark.asyncio
async def test_recommended_providers_ready_once_job_wrote_result(
    mocker: MockerFixture,
):
    stored = [{"provider": "slack", "reason": "You mentioned Slack."}, {"bad": 1}]
    mocker.patch.object(
        service.db,
        "get_dump",
        AsyncMock(return_value=_dump(recommendedProviders=stored)),
    )
    response = await service.get_recommended_providers(USER_ID)
    assert response.ready is True
    assert [p.provider for p in response.providers] == ["slack"]


@pytest.mark.asyncio
async def test_recommended_providers_ready_without_transcript(
    mocker: MockerFixture,
):
    mocker.patch.object(
        service.db,
        "get_dump",
        AsyncMock(return_value=_dump(transcript=None)),
    )
    response = await service.get_recommended_providers(USER_ID)
    assert response.ready is True
    assert response.providers == []
