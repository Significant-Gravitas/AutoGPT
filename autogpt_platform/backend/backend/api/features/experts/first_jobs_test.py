"""Tests for each hired expert's first job and the status shown beside it."""

import json
from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.api.features.experts import first_jobs
from backend.api.features.experts.first_jobs import (
    LatestExpertSession,
    compose_first_job,
    list_first_jobs,
)
from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    ExpertDayOneItem,
    ExpertSetupItem,
)

DAY_ONE = [
    ExpertDayOneItem(title="Audit your site", description="A first SEO pass"),
    ExpertDayOneItem(title="Plan the month"),
]


def _expert(**overrides) -> Expert:
    values = {
        "id": "expert-1",
        "name": "Maria",
        "avatar_url": None,
        "role": "Marketing Specialist",
        "tagline": None,
        "bio": None,
        "skills": [],
        "identity": "You are Maria.",
        "voice_preferences": "",
        "boundaries": "",
        "protected_soul_rules": list(PROTECTED_SOUL_RULES),
        "is_template": False,
        "source_template_id": "template-1",
        "is_archived": False,
        "workflows": [],
    }
    values.update(overrides)
    return Expert(**values)


def _setup_item(resolution: str = "connect", providers=None) -> ExpertSetupItem:
    return ExpertSetupItem(
        expert_id="expert-1",
        expert_name="Maria",
        expert_avatar_url=None,
        workflow_id="workflow-1",
        workflow_name="SEO Audit",
        library_agent_id="library-agent-1",
        providers=providers if providers is not None else ["google", "notion"],
        resolution=resolution,
    )


def _session(chat_status: str = "idle") -> LatestExpertSession:
    return LatestExpertSession(
        expert_id="expert-1", session_id="session-1", chat_status=chat_status
    )


def test_first_job_is_installing_while_setup_runs():
    job = compose_first_job(
        _expert(setup_status="installing"),
        DAY_ONE,
        [_setup_item()],
        _session("running"),
        "session-card",
    )

    assert job.status == "installing"
    assert job.detail == "Setting up"
    assert job.title == "Audit your site"
    assert job.description == "A first SEO pass"


def test_first_job_needs_setup_lists_providers():
    job = compose_first_job(
        _expert(),
        DAY_ONE,
        [
            _setup_item("connect", ["google"]),
            _setup_item("allow", ["notion", "google"]),
        ],
        _session("running"),
        None,
    )

    assert job.status == "needs_setup"
    assert job.providers == ["google", "notion"]
    assert job.detail == "Needs google, notion"


def test_first_job_needs_setup_when_setup_failed():
    job = compose_first_job(_expert(setup_status="failed"), DAY_ONE, [], None, None)

    assert job.status == "needs_setup"
    assert job.detail == "Setup failed"


def test_first_job_ignores_setup_items_without_a_connection_to_make():
    job = compose_first_job(
        _expert(),
        DAY_ONE,
        [_setup_item("inputs", []), _setup_item("workflow", [])],
        None,
        None,
    )

    assert job.status == "open"
    assert job.providers == []


def test_first_job_is_working_when_session_runs():
    for chat_status in ("running", "queued"):
        job = compose_first_job(
            _expert(), DAY_ONE, [], _session(chat_status), "session-card"
        )

        assert job.status == "working"
        assert job.detail == "Working on it now"
        assert job.session_id == "session-1"


def test_first_job_needs_you_when_onboarding_card_pending():
    job = compose_first_job(_expert(), DAY_ONE, [], _session(), "session-card")

    assert job.status == "needs_you"
    assert job.detail == "Waiting for your answer"
    # The link opens the chat holding the unanswered card.
    assert job.session_id == "session-card"


def test_first_job_is_ready_with_an_idle_session():
    job = compose_first_job(_expert(), DAY_ONE, [], _session(), None)

    assert job.status == "ready"
    assert job.detail == "Ready for review"
    assert job.session_id == "session-1"


def test_first_job_is_open_without_a_session():
    job = compose_first_job(
        _expert(avatar_url="https://cdn.example/maria.png"), DAY_ONE, [], None, None
    )

    assert job.status == "open"
    assert job.detail == "Not started yet"
    assert job.session_id is None
    assert job.model_dump() == {
        "expert_id": "expert-1",
        "name": "Maria",
        "role": "Marketing Specialist",
        "avatar_url": "https://cdn.example/maria.png",
        "title": "Audit your site",
        "description": "A first SEO pass",
        "status": "open",
        "detail": "Not started yet",
        "providers": [],
        "session_id": None,
    }


def test_first_job_falls_back_when_template_has_no_day_one():
    job = compose_first_job(_expert(), [], [], None, None)

    assert job.title == "Meet Maria and pick a first job"
    assert job.description == ""


@pytest.mark.asyncio
async def test_first_jobs_skip_archived_and_templates(
    mocker: pytest_mock.MockerFixture,
):
    live = _expert(id="expert-1", source_template_id="template-1")
    archived = _expert(id="expert-2", is_archived=True)
    template = _expert(id="template-9", is_template=True, source_template_id=None)
    mocker.patch.object(
        first_jobs.experts_db,
        "list_experts",
        new=AsyncMock(return_value=[live, archived, template]),
    )
    day_ones = mocker.patch.object(
        first_jobs,
        "_template_day_ones",
        new=AsyncMock(return_value={"template-1": DAY_ONE}),
    )
    mocker.patch.object(first_jobs, "list_setup_items", new=AsyncMock(return_value=[]))
    sessions = mocker.patch.object(
        first_jobs,
        "_latest_sessions",
        new=AsyncMock(return_value={"expert-1": _session()}),
    )
    onboarding = mocker.patch.object(
        first_jobs, "_pending_onboardings", new=AsyncMock(return_value={})
    )

    response = await list_first_jobs("user-1")

    assert [job.expert_id for job in response.items] == ["expert-1"]
    assert response.items[0].status == "ready"
    assert response.items[0].title == "Audit your site"
    day_ones.assert_awaited_once_with(["template-1"])
    sessions.assert_awaited_once_with("user-1", ["expert-1"])
    onboarding.assert_awaited_once_with("user-1", ["expert-1"])


@pytest.mark.asyncio
async def test_first_jobs_are_empty_without_hires(mocker: pytest_mock.MockerFixture):
    mocker.patch.object(
        first_jobs.experts_db, "list_experts", new=AsyncMock(return_value=[])
    )
    sessions = mocker.patch.object(first_jobs, "_latest_sessions", new=AsyncMock())

    response = await list_first_jobs("user-1")

    assert response.items == []
    sessions.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_onboardings_group_cards_by_expert(
    mocker: pytest_mock.MockerFixture,
):
    card = json.dumps(
        {
            "type": "expert_onboarding",
            "message": "Setup",
            "expert_id": "expert-1",
            "greeting": "I'm Maria.",
            "steps": [{"keyword": "site", "question": "Which site?", "options": []}],
        }
    )
    rows = [
        first_jobs.ExpertOnboardingMessage(
            expert_id="expert-1", session_id="session-card", role="tool", content=card
        ),
        first_jobs.ExpertOnboardingMessage(
            expert_id="expert-2", session_id="session-2", role="user", content=""
        ),
    ]
    query = mocker.patch.object(
        first_jobs, "query_raw_with_schema", new=AsyncMock(return_value=rows)
    )

    pending = await first_jobs._pending_onboardings("user-1", ["expert-1", "expert-2"])

    assert pending == {"expert-1": "session-card"}
    assert query.await_args.args[1:3] == ("user-1", ["expert-1", "expert-2"])
