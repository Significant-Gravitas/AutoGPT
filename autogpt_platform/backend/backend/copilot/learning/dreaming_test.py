from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from backend.copilot.dream import nightly_batch, orchestrator
from backend.copilot.dream.ratification import RatificationResult
from backend.copilot.dream.schemas import DreamPassResult

from .dispositions import SkillLearningResult


@pytest.mark.asyncio
async def test_dreaming_reviews_private_skills_even_when_memory_dreaming_fails(
    monkeypatch,
):
    now = datetime.now(timezone.utc)
    user = "account-a"
    learned = SkillLearningResult(
        user_id=user, run_id="learn-1", trigger="nightly", started_at=now, applied=1
    )
    run = AsyncMock(return_value=learned)
    monkeypatch.setattr(
        nightly_batch, "is_feature_enabled", AsyncMock(return_value=True), raising=False
    )
    monkeypatch.setattr(nightly_batch, "run_skill_learning_pass", run, raising=False)
    monkeypatch.setattr(
        nightly_batch, "check_dream_budget", AsyncMock(return_value=(True, None))
    )
    monkeypatch.setattr(
        orchestrator,
        "execute_dream_pass",
        AsyncMock(
            return_value=DreamPassResult(
                user_id=user,
                pass_id="dream-1",
                execution_path="sync_baseline",
                started_at=now,
                error="memory unavailable",
            )
        ),
    )
    monkeypatch.setattr(
        nightly_batch,
        "run_ratification_pass",
        AsyncMock(return_value=RatificationResult(user_id=user, started_at=now)),
    )
    result = await nightly_batch.run_nightly_batch_submit(user)
    run.assert_awaited_once_with(user, trigger="nightly")
    assert result.learning == learned
    assert result.dream.error == "memory unavailable"
