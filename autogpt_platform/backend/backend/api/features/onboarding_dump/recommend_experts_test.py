"""Tests for the expert-recommendation job.

A hire card names a person and offers to add them to the user's team, so
the filtering here is the whole safety story: only live template ids,
only that template's real workflows, only raise roles the ``/raise``
wizard can actually open. The deterministic fallback gets the same
scrutiny — it is what Path B, a flag-off run and a dead LLM all serve.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.experts.models import Expert, ExpertWorkflowRef
from backend.api.features.onboarding_dump import recommend_experts

TRANSCRIPT = "LinkedIn posts and cold outreach eat my week, and tickets pile up."


def _template(
    template_id: str, name: str, role: str, workflows: list[str] = []
) -> Expert:
    return Expert.model_construct(
        id=template_id,
        name=name,
        avatar_url=f"/experts/{name.lower()}.svg",
        role=role,
        tagline=f"{name} handles {role.lower()}.",
        bio=None,
        skills=[],
        identity="",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=[],
        is_template=True,
        source_template_id=None,
        is_archived=False,
        workflows=[
            ExpertWorkflowRef.model_construct(
                id=f"wf-{index}",
                store_listing_version_id=None,
                library_agent_id=None,
                graph_id=None,
                name=workflow,
                description=None,
            )
            for index, workflow in enumerate(workflows)
        ],
    )


MARIA = _template(
    "tpl-maria",
    "Maria",
    "Marketing",
    ["LinkedIn Post Generator", "Automated Blog Writer", "Webpage Copy Improver"],
)
MAX = _template("tpl-max", "Max", "Sales", ["Lead Finder", "Contact Enricher"])
FRANKIE = _template("tpl-frankie", "Frankie", "Ops", ["Meeting Brief"])
ROSTER = [MARIA, MAX, FRANKIE]
ELI = _template("tpl-eli", "Eli", "Engineering")
NADIA = _template("tpl-nadia", "Nadia", "Code quality")
RHEA = _template("tpl-rhea", "Rhea", "Recruiting")
SHAY = _template("tpl-shay", "Shay", "Talent sourcing")
FULL_ROSTER = [*ROSTER, ELI, NADIA, RHEA, SHAY]


def test_parse_drops_unknown_and_duplicate_templates():
    data = {
        "diagnosis": "You have a marketing problem.",
        "experts": [
            {"template_id": "tpl-maria", "reason": "You post on LinkedIn weekly."},
            {"template_id": "tpl-maria", "reason": "Duplicate."},
            {"template_id": "tpl-hallucinated", "reason": "Invented."},
            "not-a-dict",
            {"reason": "missing template_id"},
            {"template_id": "tpl-max", "reason": "Cold outreach eats your week."},
        ],
    }

    team = recommend_experts._parse_team(data, ROSTER)

    assert [e.template_id for e in team.experts] == ["tpl-maria", "tpl-max"]
    assert team.source == "llm"
    assert team.diagnosis == "You have a marketing problem."


def test_parse_copies_identity_from_the_template_not_the_model():
    data = {
        "experts": [
            {
                "template_id": "tpl-maria",
                "name": "Impostor",
                "role": "CFO",
                "avatar_url": "https://evil.example/avatar.png",
                "reason": "x" * 500,
            }
        ]
    }

    expert = recommend_experts._parse_team(data, ROSTER).experts[0]

    assert expert.name == "Maria"
    assert expert.role == "Marketing"
    assert expert.avatar_url == "/experts/maria.svg"
    assert len(expert.reason) == recommend_experts.MAX_REASON_CHARS


def test_parse_caps_the_number_of_experts_and_the_diagnosis():
    roster = [_template(f"tpl-{i}", f"Expert{i}", "Ops") for i in range(18)]
    data = {
        "diagnosis": "y" * 900,
        "experts": [{"template_id": t.id} for t in roster],
    }

    team = recommend_experts._parse_team(data, roster)

    assert len(team.experts) == recommend_experts.MAX_EXPERTS
    assert len(team.diagnosis) == recommend_experts.MAX_DIAGNOSIS_CHARS


def test_parse_keeps_only_that_templates_real_workflows():
    data = {
        "experts": [
            {
                "template_id": "tpl-maria",
                "workflow_names": [
                    "linkedin post generator",
                    "Meeting Brief",
                    "Made Up Workflow",
                    "Automated Blog Writer",
                    12,
                ],
            }
        ]
    }

    expert = recommend_experts._parse_team(data, ROSTER).experts[0]

    # Matched case-insensitively, rendered in the workflow's own spelling.
    assert expert.workflow_names == ["LinkedIn Post Generator", "Automated Blog Writer"]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "support",
        {"reason": "no role"},
        {"role": "chief happiness officer", "reason": "Not a raise role."},
    ],
)
def test_parse_drops_raise_suggestions_the_wizard_cannot_open(raw: object):
    team = recommend_experts._parse_team({"raise_suggestion": raw}, ROSTER)

    assert team.raise_suggestion is None


def test_parse_keeps_a_valid_raise_suggestion():
    team = recommend_experts._parse_team(
        {"raise_suggestion": {"role": "Support", "reason": "Tickets pile up."}}, ROSTER
    )

    assert team.raise_suggestion is not None
    assert team.raise_suggestion.role == "support"
    assert team.raise_suggestion.reason == "Tickets pile up."


@pytest.mark.parametrize(
    ("user_role", "pain_points", "expected_lead", "expected_raise"),
    [
        ("Marketing", [], ["Maria"], None),
        ("Sales/BD", [], ["Max"], None),
        ("Operations", [], ["Frankie"], None),
        ("Product/PM", [], ["Frankie"], None),
        ("Founder/CEO", [], ["Max", "Maria"], None),
        ("Engineering", [], [], "developer"),
        ("HR/People", [], ["Frankie"], "recruiter"),
        (None, ["Social media"], ["Maria"], None),
        (None, ["Finding leads"], ["Max"], None),
        (None, ["Email & outreach"], ["Max"], None),
        (None, ["Reports & data"], ["Frankie"], None),
        (None, ["Scheduling"], ["Frankie"], None),
        (None, ["CRM & data entry"], ["Frankie"], None),
        (None, ["Customer support"], [], "support"),
        (None, ["Research"], [], "researcher"),
        (
            "Marketing",
            ["Finding leads", "Scheduling"],
            ["Maria", "Max", "Frankie"],
            None,
        ),
        ("Other", ["Something else"], [], None),
    ],
)
def test_the_fallback_maps_wizard_answers_to_the_roster(
    user_role: str | None,
    pain_points: list[str],
    expected_lead: list[str],
    expected_raise: str | None,
):
    team = recommend_experts.fallback_expert_recommendations(
        user_role, pain_points, ROSTER
    )

    names = [e.name for e in team.experts]
    # The wizard's answers pick who leads; the rest of the roster fills in
    # behind them so the row is never a single card.
    assert names[: len(expected_lead)] == expected_lead
    assert sorted(names) == sorted(t.name for t in ROSTER)
    assert (team.raise_suggestion.role if team.raise_suggestion else None) == (
        expected_raise
    )
    assert team.source == "fallback"
    # Never claims to have heard anything: the fallback runs when we may
    # not have a transcript at all.
    assert team.diagnosis == recommend_experts.FALLBACK_DIAGNOSIS


@pytest.mark.parametrize(
    ("user_role", "expected_names"),
    [
        ("Engineering", ["Eli", "Nadia"]),
        ("HR/People", ["Rhea", "Shay", "Frankie"]),
    ],
)
def test_the_fallback_drops_the_raise_door_once_a_template_covers_it(
    user_role: str, expected_names: list[str]
):
    # With the engineering and people rosters seeded, "nobody covers this"
    # would be a lie: the door only opens when the role is still a gap.
    team = recommend_experts.fallback_expert_recommendations(user_role, [], FULL_ROSTER)

    names = [e.name for e in team.experts]
    assert names[: len(expected_names)] == expected_names
    assert len(names) == min(recommend_experts.MAX_EXPERTS, len(FULL_ROSTER))
    assert team.raise_suggestion is None


def test_the_fallback_skips_roles_with_no_live_template():
    # Content lags demand: the roster may not have every role seeded.
    team = recommend_experts.fallback_expert_recommendations(
        "Founder/CEO", ["Scheduling"], [MARIA]
    )

    assert [e.name for e in team.experts] == ["Maria"]


def test_the_fallback_survives_an_empty_roster():
    team = recommend_experts.fallback_expert_recommendations(
        "Engineering", ["Customer support"], []
    )

    assert team.experts == []
    # The raise door does not need a roster to open.
    assert team.raise_suggestion is not None
    assert team.raise_suggestion.role == "developer"


def test_the_fallback_reason_comes_from_the_template():
    expert = recommend_experts.fallback_expert_recommendations(
        "Marketing", [], ROSTER
    ).experts[0]

    assert expert.reason == "Maria handles marketing."
    assert expert.workflow_names == [
        "LinkedIn Post Generator",
        "Automated Blog Writer",
        "Webpage Copy Improver",
    ]


def test_the_roster_prompt_line_carries_the_workflows():
    lines = recommend_experts.roster_lines([FRANKIE])

    assert lines == (
        "- tpl-frankie: Frankie — Ops. Frankie handles ops.. "
        "Workflows: Meeting Brief"
    )


@pytest.fixture
def client(mocker: MockerFixture) -> MagicMock:
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock()
    mocker.patch(
        "backend.api.features.onboarding_dump.recommend_experts.get_openai_client",
        return_value=fake,
    )
    return fake


def _completion(payload: object) -> SimpleNamespace:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.asyncio
async def test_a_good_generation_is_marked_as_read_from_the_transcript(
    client: MagicMock,
):
    client.chat.completions.create.return_value = _completion(
        {
            "diagnosis": "You have a marketing problem and a support problem.",
            "experts": [{"template_id": "tpl-maria", "reason": "You post weekly."}],
            "raise_suggestion": {"role": "support", "reason": "Tickets pile up."},
        }
    )

    team = await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT, user_role="Marketing", pain_points=[], templates=ROSTER
    )

    assert team.source == "llm"
    assert [e.name for e in team.experts] == ["Maria"]
    assert team.raise_suggestion is not None


@pytest.mark.asyncio
async def test_a_dead_generation_degrades_to_the_fallback(client: MagicMock):
    client.chat.completions.create.side_effect = RuntimeError("llm down")

    team = await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT, user_role="Sales/BD", pain_points=[], templates=ROSTER
    )

    assert team.source == "fallback"
    assert team.experts[0].name == "Max"
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_a_generation_with_nothing_usable_degrades_to_the_fallback(
    client: MagicMock,
):
    # Every id hallucinated and no raise door: there is nothing to render.
    client.chat.completions.create.return_value = _completion(
        {"experts": [{"template_id": "tpl-nope"}], "raise_suggestion": None}
    )

    team = await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT, user_role="Operations", pain_points=[], templates=ROSTER
    )

    assert team.source == "fallback"
    assert team.experts[0].name == "Frankie"


@pytest.mark.asyncio
async def test_non_json_output_is_retried_once_then_falls_back(client: MagicMock):
    client.chat.completions.create.return_value = _completion("no json here")

    team = await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT, user_role="Marketing", pain_points=[], templates=ROSTER
    )

    assert team.source == "fallback"
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_no_transcript_never_reaches_the_model(client: MagicMock):
    team = await recommend_experts.generate_expert_recommendations(
        "   ", user_role="Marketing", pain_points=[], templates=ROSTER
    )

    assert team.source == "fallback"
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_missing_llm_client_degrades_to_the_fallback(mocker: MockerFixture):
    mocker.patch(
        "backend.api.features.onboarding_dump.recommend_experts.get_openai_client",
        return_value=None,
    )

    team = await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT, user_role="Marketing", pain_points=[], templates=ROSTER
    )

    assert team.source == "fallback"


@pytest.mark.asyncio
async def test_the_prompt_carries_the_roster_and_the_wizard_answers(
    client: MagicMock,
):
    client.chat.completions.create.return_value = _completion(
        {"experts": [{"template_id": "tpl-max"}]}
    )

    await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT,
        user_role="Founder/CEO",
        pain_points=["Finding leads"],
        templates=ROSTER,
    )

    content = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert "- tpl-max: Max — Sales." in content
    assert "Lead Finder, Contact Enricher" in content
    assert "Founder/CEO" in content
    assert "Finding leads" in content
    assert TRANSCRIPT in content


def test_fallback_prioritizes_all_matching_templates_before_truncating():
    unrelated = [_template(f"other-{i}", f"Other{i}", "Ops") for i in range(16)]
    first = _template("sales-first", "First", "Sales")
    second = _template("sales-second", "Second", " SALES ")
    team = recommend_experts.fallback_expert_recommendations(
        "Sales/BD", ["Finding leads", "Email & outreach"], [*unrelated, first, second]
    )
    ids = [expert.template_id for expert in team.experts]
    assert ids[:2] == [first.id, second.id]
    assert len(ids) == len(set(ids)) == recommend_experts.MAX_EXPERTS


@pytest.mark.asyncio
async def test_prompt_treats_braces_in_user_and_roster_text_as_literal(
    client: MagicMock,
):
    client.chat.completions.create.return_value = _completion(
        {"experts": [{"template_id": "tpl-max"}]}
    )
    transcript = (
        'I maintain JSON like {"customer": "{name}"} and {missing} placeholders.'
    )
    template = MAX.model_copy(update={"tagline": "Handles {sales} workflows"})
    team = await recommend_experts.generate_expert_recommendations(
        transcript,
        user_role="{Founder}",
        pain_points=["{Reports}"],
        templates=[template],
    )
    content = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert transcript in content
    assert "{Founder}" in content
    assert "{Reports}" in content
    assert "{sales}" in content
    assert f"at most {recommend_experts.MAX_EXPERTS} objects" in content
    assert "all three slots" not in content
    assert team.source == "llm"
