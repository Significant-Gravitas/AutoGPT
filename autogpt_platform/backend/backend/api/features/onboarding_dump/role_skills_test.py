"""Work matched to the live roster by the words of its roles: the plan's task
owners and the role fallback team, against the roles the skills catalog
ships (Significant-Gravitas/skills-catalog, experts/*.yml)."""

import pytest

from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump import (
    plan_owners,
    recommend_experts,
    role_skills,
    role_task_sets,
    role_tasks,
)
from backend.api.features.onboarding_dump.models import (
    RecommendedExpert,
    SuggestedPrompt,
)

# The catalog's roster, in its order: each template's name and role.
LIVE_ROSTER = [
    ("Alex", "Product"),
    ("Anika", "Partnerships"),
    ("Casey", "Code Review & QA"),
    ("Clip", "Motion Graphics and Video Creation"),
    ("Daniel", "Finance"),
    ("Devon", "Dependency & Security Hygiene"),
    ("Ellis", "Contracts"),
    ("Frankie", "Ops"),
    ("Harper", "Recruiting & Hiring"),
    ("Ines", "People Ops & HR"),
    ("James", "Operations"),
    ("Jordan", "Deal Desk & Proposal Support"),
    ("Jules", "Social & Content Repurposing"),
    ("Kai", "Executive Assistant"),
    ("Lena", "Privacy & Compliance"),
    ("Marco", "Paid Ads & Performance"),
    ("Maria", "SEO & Content"),
    ("Max", "Sales"),
    ("Maya", "Marketing"),
    ("Mina", "Finance, Invoicing & Bookkeeping"),
    ("Nadia", "Market & Competitor Intelligence"),
    ("Noor", "PR & Communications"),
    ("Omar", "RevOps & CRM Hygiene"),
    ("Priya", "Product Management"),
    ("Quinn", "Research, Data & KPI Analysis"),
    ("Remy", "Email & Lifecycle"),
    ("Riley", "Customer Success & Retention"),
    ("Robin", "Customer Support"),
    ("Sasha", "Support & Help Desk"),
    ("Sofia", "Recruiting"),
    ("Theo", "Finance, Fundraising & Investor Relations"),
    ("Vera", "Vendor & Procurement"),
    ("Zara", "Go-to-Market"),
]


def _template(name: str, role: str) -> Expert:
    return Expert.model_construct(
        id=f"tpl-{name.lower()}",
        name=name,
        avatar_url=None,
        role=role,
        tagline=f"{name} handles {role.lower()}.",
        workflows=[],
    )


def _expert(name: str) -> RecommendedExpert:
    return RecommendedExpert(
        template_id=f"tpl-{name.lower()}", name=name, role=dict(LIVE_ROSTER)[name]
    )


TEMPLATES = [_template(name, role) for name, role in LIVE_ROSTER]


def _names(template_ids: list[str | None]) -> list[str | None]:
    return [tid.removeprefix("tpl-").title() if tid else None for tid in template_ids]


def test_a_product_plan_with_a_product_team_names_them_not_otto():
    # A product manager's brain dump: the team job picked Nadia, Alex and
    # Quinn, and the greeting fell back to the Product/PM set.
    team = [_expert("Nadia"), _expert("Alex"), _expert("Quinn")]

    prompts = role_tasks.role_prompts("Product/PM", team)

    assert _names([p.expert_template_id for p in prompts]) == [
        "Alex",  # customer feedback, ranked
        "Alex",  # the spec
        "Nadia",  # what competitors shipped
        "Quinn",  # the Monday metrics summary
        "Alex",  # release notes
    ]
    assert [p.expert_name for p in prompts][:3] == ["Alex", "Alex", "Nadia"]


def test_a_product_manager_with_no_brain_dump_gets_a_product_team():
    plan, team = role_tasks.role_plan("Product/PM", [], TEMPLATES)

    assert [e.name for e in team] == ["Alex", "Quinn", "Nadia"]
    assert _names([p.expert_template_id for p in plan]) == [
        "Alex",
        "Alex",
        "Nadia",
        "Quinn",
        "Alex",
    ]


@pytest.mark.parametrize(
    ("user_role", "pain_points", "expected"),
    [
        ("Founder/CEO", [], ["Max", "Maya", "Daniel"]),
        ("Marketing", [], ["Maya", "Nadia", "Quinn"]),
        ("Sales/BD", [], ["Max", "Omar"]),
        ("Operations", [], ["Frankie", "Kai"]),
        ("Product/PM", [], ["Alex", "Quinn", "Nadia"]),
        ("Engineering", [], ["Casey", "Devon", "Quinn"]),
        # Ines (People Ops & HR) covers ops too, so nobody else is added.
        ("HR/People", [], ["Sofia", "Ines"]),
        ("Other", ["Reports & data"], ["Quinn"]),
        ("Other", ["CRM & data entry"], ["Omar"]),
        ("Other", ["Customer support"], ["Robin"]),
        ("Other", ["Research"], ["Quinn"]),
        # The need the user named comes before finance, which the role only
        # suggests.
        ("Founder/CEO", ["Scheduling", "Finding leads"], ["Max", "Kai", "Maya"]),
    ],
)
def test_the_fallback_team_is_the_roster_s_best_fit_for_each_need(
    user_role: str, pain_points: list[str], expected: list[str]
):
    team = recommend_experts.fallback_expert_recommendations(
        user_role, pain_points, TEMPLATES
    )

    assert [e.name for e in team.experts] == expected


def test_a_need_the_team_already_covers_adds_nobody():
    # Quinn, picked for the data, also covers research.
    team = recommend_experts.fallback_expert_recommendations(
        "Product/PM", ["Research"], TEMPLATES
    )

    assert [e.name for e in team.experts] == ["Alex", "Quinn", "Nadia"]
    assert team.raise_suggestion is None


def test_a_need_the_roster_now_covers_closes_the_raise_door():
    team = recommend_experts.fallback_expert_recommendations(
        "Other", ["Customer support"], TEMPLATES
    )

    assert team.raise_suggestion is None


@pytest.mark.parametrize("user_role", sorted(role_task_sets.ROLE_FALLBACK_TASKS))
def test_every_role_plan_is_the_team_s_work_not_otto_s(user_role: str):
    plan, team = role_tasks.role_plan(user_role, [], TEMPLATES)
    owners = [p.expert_template_id for p in plan]

    if user_role == role_task_sets.OTHER_ROLE:
        # No role, no team: Otto takes it all.
        assert team == [] and owners == [None] * len(plan)
        return
    assert owners[0] in {e.template_id for e in team}
    # Engineering's customer questions are the one task no engineering
    # expert covers.
    assert owners.count(None) <= 1


def test_a_task_whose_best_match_is_cut_from_the_team_goes_to_one_who_stays():
    # Four recommended, three kept: the first three tasks keep Alex, Quinn
    # and Maya, so Max, the best match for the leads, is not hired.
    team = [_expert("Maya"), _expert("Alex"), _expert("Quinn"), _expert("Max")]
    prompts = [
        SuggestedPrompt(title="Draft the PRD", prompt=".", icon="file-text"),
        SuggestedPrompt(title="Track the funnel", prompt=".", icon="chart-bar"),
        SuggestedPrompt(title="Write the launch posts", prompt=".", icon="megaphone"),
        SuggestedPrompt(
            title="Announce the launch to your leads", prompt=".", icon="megaphone"
        ),
    ]

    owned, kept = plan_owners.cap_team(plan_owners.fallback_assign(prompts, team), team)

    assert [e.name for e in kept] == ["Alex", "Quinn", "Maya"]
    # The announcement is marketing too, so it is Maya's, not Otto's.
    assert _names([p.expert_template_id for p in owned]) == [
        "Alex",
        "Quinn",
        "Maya",
        "Maya",
    ]


@pytest.mark.parametrize(
    ("role", "skills"),
    [
        ("Research, Data & KPI Analysis", {"data", "research"}),
        ("Market & Competitor Intelligence", {"competitors"}),
        ("Product", {"product"}),
        ("Dependency & Security Hygiene", {"security"}),
        ("Code Review & QA", {"engineering"}),
        ("RevOps & CRM Hygiene", {"sales", "crm"}),
        ("People Ops & HR", {"people", "ops"}),
        ("PR & Communications", {"marketing"}),
        ("Deal Desk & Proposal Support", {"sales"}),
        ("Go-to-Market", {"marketing"}),
        ("  EXECUTIVE   assistant ", {"assistant"}),
        ("Astrologer", set()),
    ],
)
def test_a_role_s_skills_come_from_its_whole_words_in_any_case(
    role: str, skills: set[str]
):
    assert role_skills.skills_of(role) == skills


def test_a_greeting_task_with_no_jev_goes_to_whoever_its_title_names():
    team = [_expert("Nadia"), _expert("Alex"), _expert("Quinn")]
    prompts = [
        SuggestedPrompt(
            title="Draft the PRD for bulk export", prompt=".", icon="file-text"
        ),
        SuggestedPrompt(
            title="Track the activation funnel weekly", prompt=".", icon="chart-bar"
        ),
        SuggestedPrompt(title="Watch what competitors ship", prompt=".", icon="bell"),
        SuggestedPrompt(title="Plan your week", prompt=".", icon="calendar-check"),
    ]

    owned = plan_owners.fallback_assign(prompts, team)

    assert _names([p.expert_template_id for p in owned]) == [
        "Alex",
        "Quinn",
        "Nadia",
        None,
    ]


def test_the_most_focused_template_takes_a_need():
    # "Sales" over "Deal Desk & Proposal Support"; "Product" over "Product
    # Management"; among equals, the roster's order.
    assert role_skills.best_template("sales", TEMPLATES).name == "Max"
    assert role_skills.best_template("product", TEMPLATES).name == "Alex"
    assert role_skills.best_template("legal", []) is None
