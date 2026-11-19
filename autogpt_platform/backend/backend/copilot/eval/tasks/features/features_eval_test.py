"""Tests for the task features eval's cases and checks, on fakes: the case
files and suggestion drafting on an in-memory account."""

import pytest

from backend.copilot.briefing import suggestions
from backend.copilot.eval.tasks.__main__ import load_worlds
from backend.copilot.eval.tasks.models import CheckResult

from .__main__ import load_cases
from .fakes import FakeBackend
from .models import AccountSpec, FeatureChecks
from .suggestion_checks import reworded, suggestion_checks
from .suggestion_run import Drafter, draft_for

WORLDS = load_worlds()


def _failed(checks: list[CheckResult]) -> list[str]:
    return [c.name for c in checks if not c.passed]


# -- cases ---------------------------------------------------------------------


def test_case_files_are_consistent():
    cases = load_cases()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    assert 20 <= len(cases) <= 32
    for case in cases:
        assert case.world in WORLDS, case.id
        assert case.id.startswith(case.area), case.id
        assert case.expected.must, f"{case.id} states no expectation"
        if case.needs == "live":
            assert case.live and case.task, case.id
        elif case.area == "suggestions":
            assert case.account is not None, case.id
        owners = {str(e["id"]) for e in WORLDS[case.world].experts}
        spec_owner = case.task.owner if case.task else None
        assert spec_owner in (None, *owners), case.id


def test_case_filters():
    assert {c.area for c in load_cases(areas=["suggestions"])} == {"suggestions"}
    assert {c.needs for c in load_cases(needs=["none"])} == {"none"}
    assert [c.id for c in load_cases(ids=["controls_004"])] == ["controls_004"]


# -- suggestions -------------------------------------------------------------------


async def test_empty_account_drafts_nothing_without_a_call():
    drafting = await draft_for(AccountSpec(), [], Drafter(scripted=[]))
    assert drafting.calls == [] and drafting.returned == []


async def test_drafting_runs_production_code_and_restores_it():
    original = suggestions.task_db
    account = AccountSpec.model_validate(
        {
            "board": [
                {
                    "title": "Fix the pump supplier",
                    "status": "FAILED",
                    "failure_reason": "no reply",
                }
            ],
            "live": [{"title": "Email waitlist", "status": "TODO", "days_ago": 1}],
            "dismissed": ["Chase the pump supplier"],
        }
    )
    scripted = [
        {"title": "Chase the pump supplier", "brief": "x" * 80, "reason": "pump"},
        {
            "title": "Find a second pump supplier",
            "brief": "y" * 80,
            "reason": "pump supplier",
            "owner_expert_id": "exp_marketing",
        },
        {
            "title": "Price airless pumps",
            "brief": "z" * 80,
            "reason": "pump",
            "owner_expert_id": "nope",
        },
        {"title": "Third", "brief": "w" * 80, "reason": "r"},
    ]
    experts = WORLDS["lumen_skincare"].experts
    drafting = await draft_for(account, experts, Drafter(scripted=scripted))
    assert suggestions.task_db is original
    assert drafting.calls[0].room == 2
    assert "Chase the pump supplier" in drafting.calls[0].facts
    assert [t.title for t in drafting.created] == [
        "Find a second pump supplier",
        "Price airless pumps",
    ]
    assert [t.owner_expert_id for t in drafting.created] == ["exp_marketing", None]
    assert len(drafting.returned) == 3
    checks = suggestion_checks(
        FeatureChecks(
            no_repeats=True,
            owners_valid=True,
            max_new=2,
            reasons_grounded=True,
            model_called=True,
        ),
        drafting,
    )
    assert _failed(checks) == []


async def test_model_drafter_parses_backend_json():
    backend = FakeBackend(
        'ok {"suggestions": [{"title": "Rebook the dairy call", "brief": "Call '
        'Kisan Fresh about the Dec rate and log the outcome.", "reason": "dairy"}]}'
    )
    account = AccountSpec.model_validate(
        {"board": [{"title": "Dairy offers", "status": "DONE", "result": "done"}]}
    )
    drafting = await draft_for(account, [], Drafter(backend, "m"))
    assert [t.title for t in drafting.created] == ["Rebook the dairy call"]
    assert "<task_facts>" in backend.sent[0]


async def test_model_case_without_a_backend_is_an_error():
    account = AccountSpec.model_validate({"board": [{"title": "A", "status": "DONE"}]})
    with pytest.raises(RuntimeError):
        await draft_for(account, [], Drafter())


@pytest.mark.parametrize(
    "a,b,same",
    [
        (
            "Work out the paneer price impact across dairy supplier offers",
            "Compare dairy supplier offers and cost impact of paneer prices",
            True,
        ),
        (
            "Get a revised paneer rate from Kisan Fresh",
            "Chase Kisan Fresh for a revised paneer rate",
            True,
        ),
        (
            "Book a pest control visit for Bandra",
            "Ask Aarey Pure for a paneer quote",
            False,
        ),
        ("", "Anything", False),
    ],
)
def test_reworded(a: str, b: str, same: bool):
    assert reworded(a, b) is same


async def test_retired_suggestion_counts_as_set_aside():
    account = AccountSpec.model_validate(
        {
            "board": [{"title": "Binder", "status": "IN_PROGRESS"}],
            "live": [
                {
                    "title": "Book the pest control visit",
                    "status": "TODO",
                    "days_ago": 4,
                }
            ],
            "retired": ["Get the water test report"],
        }
    )
    scripted = [
        {
            "title": "New one",
            "brief": "b" * 80,
            "reason": "binder",
            "done_when": "Filed",
        }
    ]
    drafting = await draft_for(account, [], Drafter(scripted=scripted))
    assert drafting.calls[0].room == 3  # the stale one was retired
    assert drafting.dismissed == [
        "Get the water test report",
        "Book the pest control visit",
    ]
    assert "Book the pest control visit" in drafting.calls[0].facts
    checks = suggestion_checks(FeatureChecks(has_done_when=True), drafting)
    assert _failed(checks) == []
    drafting.created[0] = drafting.created[0].model_copy(update={"done_when": None})
    assert "has_done_when" in _failed(
        suggestion_checks(FeatureChecks(has_done_when=True), drafting)
    )
