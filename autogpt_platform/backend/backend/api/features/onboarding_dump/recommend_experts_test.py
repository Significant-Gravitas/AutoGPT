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


# This roster has one expert per skill (Maria marketing, Max sales, Frankie
# ops), so every team is all three: what matters is who leads. The experts
# the answers point at come first; generalists fill the row.
@pytest.mark.parametrize(
    ("user_role", "pain_points", "expected_team", "expected_raise"),
    [
        ("Marketing", [], ["Maria", "Max", "Frankie"], None),
        ("Sales/BD", [], ["Max", "Maria", "Frankie"], None),
        ("Operations", [], ["Frankie", "Maria", "Max"], None),
        # Nobody here does product, data or competitors: generalists only.
        ("Product/PM", [], ["Maria", "Max", "Frankie"], None),
        ("Founder/CEO", [], ["Max", "Maria", "Frankie"], None),
        ("Engineering", [], ["Maria", "Max", "Frankie"], "developer"),
        ("HR/People", [], ["Maria", "Max", "Frankie"], "recruiter"),
        (None, ["Social media"], ["Maria", "Max", "Frankie"], None),
        (None, ["Finding leads"], ["Max", "Maria", "Frankie"], None),
        (None, ["Email & outreach"], ["Max", "Maria", "Frankie"], None),
        (None, ["Reports & data"], ["Maria", "Max", "Frankie"], None),
        (None, ["Scheduling"], ["Maria", "Max", "Frankie"], None),
        (None, ["CRM & data entry"], ["Maria", "Max", "Frankie"], None),
        (None, ["Customer support"], ["Maria", "Max", "Frankie"], "support"),
        (None, ["Research"], ["Maria", "Max", "Frankie"], "researcher"),
        (
            "Marketing",
            ["Finding leads", "Scheduling"],
            ["Maria", "Max", "Frankie"],
            None,
        ),
        # The user's pain point comes before what the role only suggests.
        (
            "Operations",
            ["Finding leads", "Scheduling"],
            ["Frankie", "Max", "Maria"],
            None,
        ),
        ("Other", ["Something else"], ["Maria", "Max", "Frankie"], None),
    ],
)
def test_the_fallback_maps_wizard_answers_to_the_roster(
    user_role: str | None,
    pain_points: list[str],
    expected_team: list[str],
    expected_raise: str | None,
):
    team = recommend_experts.fallback_expert_recommendations(
        user_role, pain_points, ROSTER
    )

    assert [e.name for e in team.experts] == expected_team
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
        # One template per skill: Eli and Rhea, the most focused, take
        # engineering and recruiting over Nadia and Shay; generalists fill
        # the row.
        ("Engineering", ["Eli", "Maria", "Max"]),
        ("HR/People", ["Rhea", "Maria", "Max"]),
    ],
)
def test_the_fallback_drops_the_raise_door_once_a_template_covers_it(
    user_role: str, expected_names: list[str]
):
    # With the engineering and people rosters seeded, "nobody covers this"
    # would be a lie: the door only opens when the role is still a gap.
    team = recommend_experts.fallback_expert_recommendations(user_role, [], FULL_ROSTER)

    assert [e.name for e in team.experts] == expected_names
    assert team.raise_suggestion is None


def test_the_fallback_skips_roles_with_no_live_template():
    # Content lags demand: the roster may not have every role seeded, and a
    # team is never padded with someone whose role names no skill.
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


def _completion(payload: object, finish_reason: str = "stop") -> SimpleNamespace:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ]
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
    # The model's pick first, then the wizard's, so the user starts with a
    # team of three.
    assert [e.name for e in team.experts] == ["Maria", "Max", "Frankie"]
    assert team.experts[0].reason == "You post weekly."
    assert team.raise_suggestion is not None


@pytest.mark.asyncio
async def test_a_full_generation_is_kept_as_the_model_ranked_it(client: MagicMock):
    client.chat.completions.create.return_value = _completion(
        {
            "experts": [
                {"template_id": "tpl-frankie"},
                {"template_id": "tpl-max"},
                {"template_id": "tpl-maria"},
            ]
        }
    )

    team = await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT, user_role="Marketing", pain_points=[], templates=ROSTER
    )

    assert [e.name for e in team.experts] == ["Frankie", "Max", "Maria"]


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
async def test_non_json_output_is_retried_once_then_falls_back(
    client: MagicMock, caplog: pytest.LogCaptureFixture
):
    client.chat.completions.create.return_value = _completion(
        '{"diagnosis": "You said tickets pile', finish_reason="length"
    )

    team = await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT, user_role="Marketing", pain_points=[], templates=ROSTER
    )

    assert team.source == "fallback"
    assert client.chat.completions.create.await_count == 2
    # Why it failed, never what the model wrote: the output echoes the dump.
    assert "finish_reason=length, 36 chars" in caplog.text
    assert "tickets pile" not in caplog.text


@pytest.mark.asyncio
async def test_the_generation_asks_for_no_reasoning(client: MagicMock):
    # Haiku 5.5's adaptive thinking spent the whole output budget on a real
    # brain dump and returned no JSON; the answer is a matching task.
    client.chat.completions.create.return_value = _completion(
        {"experts": [{"template_id": "tpl-max"}]}
    )

    await recommend_experts.generate_expert_recommendations(
        TRANSCRIPT, user_role="Sales/BD", pain_points=[], templates=ROSTER
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["extra_body"] == {"reasoning": {"enabled": False}}


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


def test_fallback_fills_one_row_and_never_with_roles_that_name_no_skill():
    unrelated = [_template(f"other-{i}", f"Other{i}", "Astrology") for i in range(16)]
    sales = [_template(f"sales-{i}", f"Seller{i}", " SALES ") for i in range(5)]
    needed = [
        *sales,
        _template("tpl-helper", "Helper", "Executive Assistant"),
        _template("tpl-analyst", "Analyst", "Data"),
        _template("tpl-writer", "Writer", "Marketing"),
    ]
    roster = [*unrelated, *needed]

    # One template per need, never a second seller for the same work; the
    # rest of the row is generalists.
    team = recommend_experts.fallback_expert_recommendations(
        "Sales/BD", ["Finding leads", "Email & outreach"], roster
    )
    assert [expert.template_id for expert in team.experts] == [
        "sales-0",
        "tpl-helper",
        "tpl-writer",
    ]

    # The role's first need, then the user's own, up to one row: marketing
    # and finance, inferred from the role, are cut before them.
    team = recommend_experts.fallback_expert_recommendations(
        "Founder/CEO", ["Scheduling", "Reports & data"], roster
    )
    assert [expert.name for expert in team.experts] == ["Seller0", "Helper", "Analyst"]
    assert len(team.experts) == recommend_experts.MAX_FALLBACK_EXPERTS

    team = recommend_experts.fallback_expert_recommendations("Other", [], roster)
    assert [expert.name for expert in team.experts] == ["Writer", "Helper", "Seller0"]

    team = recommend_experts.fallback_expert_recommendations("Other", [], unrelated)
    assert team.experts == []


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
