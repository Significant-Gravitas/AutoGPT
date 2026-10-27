"""Who takes each task when Jev does not say, and the team the plan shows:
the hinted owners, the team cap and the role starter sets."""

import pytest

from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump import (
    intro,
    plan_owners,
    recommend_experts,
    role_task_sets,
    role_tasks,
    suggested_tasks,
)
from backend.api.features.onboarding_dump.models import (
    MAX_WHY_CHARS,
    RecommendedExpert,
    SuggestedPrompt,
)
from backend.data.onboarding_role import OTHER, ROLE_LABELS

MARIA = RecommendedExpert(template_id="tpl-maria", name="Maria", role="Marketing")
MAX = RecommendedExpert(template_id="tpl-max", name="Max", role="Sales")
MINA = RecommendedExpert(template_id="tpl-mina", name="Mina", role="Bookkeeping")
NADIA = RecommendedExpert(template_id="tpl-nadia", name="Nadia", role="Research")
OLIVE = RecommendedExpert(template_id="tpl-olive", name="Olive", role="Ops")
TEAM = [MARIA, MAX, MINA, NADIA]

PROMPTS = [
    SuggestedPrompt(title="Rewrite your Etsy listings", prompt="Rewrite them."),
    SuggestedPrompt(title="Find 20 gift shops", prompt="Find shops."),
    SuggestedPrompt(title="Track orders and costs", prompt="Track them."),
    SuggestedPrompt(title="Plan a holiday drop", prompt="Plan it."),
]

# The wizard's option ids, then Other, which also stands for its typed text.
WIZARD_ROLES = [*ROLE_LABELS, role_task_sets.OTHER_ROLE]
# The roster templates' roles, lower-cased as the owner match reads them:
# every role the fallback team is built from (``recommend_experts``) and
# Bookkeeping, which only the model's recommendation picks.
TEMPLATE_ROLES = {
    "sales",
    "marketing",
    "ops",
    "office ops",
    "executive assistant",
    "project management",
    "engineering",
    "code quality",
    "recruiting",
    "talent sourcing",
    "bookkeeping",
}
# Roles the owner hints name ahead of the roster: no template has one today,
# so the hint gives no owner until one does (``plan_owners.OWNER_HINTS``).
# The role starter sets name only roles the roster has.
NOT_ON_THE_ROSTER_YET = {"research"}


def _hinted_roles() -> list[tuple[str, ...]]:
    return [*plan_owners.OWNER_HINTS.values()] + [
        roles for _, roles in plan_owners.TITLE_HINTS
    ]


def test_the_starter_sets_cover_every_role_the_wizard_offers_and_other():
    assert role_task_sets.OTHER_ROLE == OTHER
    assert sorted(role_task_sets.ROLE_FALLBACK_TASKS) == sorted(WIZARD_ROLES)


def test_every_role_the_fallback_team_is_built_from_is_a_template_role():
    wanted = [
        *recommend_experts._ROLE_TEMPLATE_ROLES.values(),
        *recommend_experts._PAIN_TEMPLATE_ROLES.values(),
    ]

    assert {role for roles in wanted for role in roles} <= TEMPLATE_ROLES


def test_every_owner_hint_names_a_prompt_icon_and_template_roles():
    assert set(plan_owners.OWNER_HINTS) <= intro.PROMPT_ICONS
    for roles in _hinted_roles():
        assert set(roles) <= TEMPLATE_ROLES | NOT_ON_THE_ROSTER_YET, roles


def test_a_hint_for_a_role_not_on_the_roster_yet_falls_back_to_one_that_is():
    for roles in _hinted_roles():
        if NOT_ON_THE_ROSTER_YET & set(roles):
            assert TEMPLATE_ROLES & set(roles), roles


def test_every_wizard_role_has_five_starter_tasks():
    for role, tasks in role_task_sets.ROLE_FALLBACK_TASKS.items():
        assert len(tasks) == 5, role
        assert tasks[0].why, f"{role}'s first task explains itself"


def test_role_tasks_are_valid_copy_owned_by_roles_the_roster_has():
    for tasks in role_task_sets.ROLE_FALLBACK_TASKS.values():
        for task in tasks:
            assert task.icon in intro.PROMPT_ICONS, task.title
            assert task.owner_role is None or task.owner_role in TEMPLATE_ROLES
            assert task.why is None or len(task.why) <= MAX_WHY_CHARS
            assert len(task.title) <= intro.MAX_TITLE_CHARS, task.title
            for text in (task.title, task.prompt, task.why or ""):
                assert "—" not in text and "–" not in text, text


def test_the_fallback_keeps_the_order_and_matches_owners_by_icon_and_title():
    prompts = [
        SuggestedPrompt(title="Chase cold leads", prompt=".", icon="sparkle"),
        SuggestedPrompt(title="Draft the newsletter", prompt=".", icon="megaphone"),
        SuggestedPrompt(title="Reconcile the books", prompt=".", icon="chart-bar"),
        SuggestedPrompt(title="Watch competitors", prompt=".", icon="bell"),
        SuggestedPrompt(title="Fix the deploy script", prompt=".", icon="code"),
        SuggestedPrompt(title="Plan a team offsite", prompt=".", icon="sparkle"),
    ]

    assigned = plan_owners.fallback_assign(prompts, TEAM)

    assert [p.title for p in assigned] == [p.title for p in prompts]
    assert [p.expert_template_id for p in assigned] == [
        "tpl-max",  # title: leads
        "tpl-maria",  # icon: megaphone
        "tpl-mina",  # icon: chart-bar, bookkeeping before ops
        "tpl-nadia",  # icon: bell, research before marketing
        None,  # code, but nobody on the team does engineering
        None,  # no hint at all
    ]


def test_the_icon_s_hint_comes_before_the_title_s():
    prompt = SuggestedPrompt(
        title="Announce the launch to your leads", prompt=".", icon="megaphone"
    )

    [assigned] = plan_owners.fallback_assign([prompt], TEAM)

    assert assigned.expert_template_id == "tpl-maria"


def test_a_hinted_role_nobody_on_the_team_has_falls_to_the_next_one():
    prompts = [
        SuggestedPrompt(title="Weekly numbers", prompt=".", icon="chart-bar"),
        SuggestedPrompt(title="Watch competitors", prompt=".", icon="bell"),
    ]

    assigned = plan_owners.fallback_assign(prompts, [MARIA, OLIVE])

    # Bookkeeping and research are not on this team; ops and marketing are.
    assert [p.expert_template_id for p in assigned] == ["tpl-olive", "tpl-maria"]


def test_an_owner_is_matched_by_role_whatever_its_case_and_spacing():
    team = [
        RecommendedExpert(
            template_id="tpl-eve", name="Eve", role="  Executive Assistant "
        ),
        RecommendedExpert(template_id="tpl-sam", name="Sam", role="SALES"),
        RecommendedExpert(template_id="tpl-sid", name="Sid", role="Sales"),
    ]

    assert plan_owners.owner_for_role("executive assistant", team) == "tpl-eve"
    # Two people with the role: the first on the team takes it.
    assert plan_owners.owner_for_role("sales", team) == "tpl-sam"
    assert plan_owners.owner_for_role("engineering", team) is None
    assert plan_owners.owner_for_role("sales", []) is None


def test_the_team_is_the_first_three_tasks_owners_then_the_recommendation():
    prompts = [
        PROMPTS[0].model_copy(update={"expert_template_id": "tpl-nadia"}),
        PROMPTS[1].model_copy(update={"expert_template_id": None}),
        PROMPTS[2].model_copy(update={"expert_template_id": "tpl-nadia"}),
        PROMPTS[3].model_copy(update={"expert_template_id": "tpl-mina"}),
    ]

    capped_prompts, team = plan_owners.cap_team(prompts, TEAM)

    # Nadia owns a top-three task; Maria and Max pad from the recommendation.
    assert [e.name for e in team] == ["Nadia", "Maria", "Max"]
    # Mina is not on the capped team, so her task goes to Otto.
    assert [p.expert_template_id for p in capped_prompts] == [
        "tpl-nadia",
        None,
        "tpl-nadia",
        None,
    ]


def test_a_team_of_four_owners_keeps_three_and_nulls_the_fourth():
    prompts = [
        p.model_copy(update={"expert_template_id": e.template_id})
        for p, e in zip(PROMPTS, [MINA, MAX, MARIA, NADIA])
    ]

    capped_prompts, team = plan_owners.cap_team(prompts, TEAM)

    assert [e.name for e in team] == ["Mina", "Max", "Maria"]
    assert len(team) == plan_owners.MAX_TEAM
    assert capped_prompts[3].expert_template_id is None


def test_a_kept_owner_s_name_is_written_beside_them_and_a_dropped_one_s_cleared():
    # Home names whoever a task was for even after they leave the team.
    prompts = [
        p.model_copy(update={"expert_template_id": e.template_id})
        for p, e in zip(PROMPTS, [MINA, MAX, MARIA, NADIA])
    ]
    prompts[3] = prompts[3].model_copy(update={"expert_name": "Nadia"})

    capped_prompts, _ = plan_owners.cap_team(prompts, TEAM)

    assert [p.expert_name for p in capped_prompts] == ["Mina", "Max", "Maria", None]


def test_an_owner_who_was_never_recommended_is_not_shown_and_loses_the_task():
    prompts = [PROMPTS[0].model_copy(update={"expert_template_id": "tpl-ghost"})]

    capped_prompts, team = plan_owners.cap_team(prompts, [MARIA])

    assert [e.name for e in team] == ["Maria"]
    assert capped_prompts[0].expert_template_id is None


def test_no_recommended_experts_leaves_every_task_with_otto():
    prompts = [PROMPTS[0].model_copy(update={"expert_template_id": "tpl-maria"})]

    capped_prompts, team = plan_owners.cap_team(prompts, [])

    assert team == []
    assert capped_prompts[0].expert_template_id is None


def _template(template_id: str, name: str, role: str) -> Expert:
    return Expert.model_construct(
        id=template_id,
        name=name,
        avatar_url=None,
        role=role,
        tagline=f"{name} handles {role.lower()}.",
        workflows=[],
    )


ROSTER = [
    _template("tpl-max", "Max", "Sales"),
    _template("tpl-maria", "Maria", "Marketing"),
    _template("tpl-olive", "Olive", "Ops"),
]


def test_a_founder_plan_takes_owners_from_the_role_team():
    prompts, team = role_tasks.role_plan("Founder/CEO", [], ROSTER)

    assert [e.name for e in team] == ["Max", "Maria"]
    assert prompts[0].title.startswith("Build a list of 25 companies")
    assert prompts[0].why == "Most founders start here."
    assert [p.expert_template_id for p in prompts] == [
        "tpl-max",
        "tpl-maria",
        None,
        None,
        "tpl-maria",
    ]
    assert [p.expert_name for p in prompts] == ["Max", "Maria", None, None, "Maria"]


def test_a_task_whose_role_is_outside_the_role_team_is_otto_s():
    # Ops is on the roster but not on a marketer's fallback team.
    prompts, team = role_tasks.role_plan("Marketing", [], ROSTER)

    assert [e.name for e in team] == ["Maria"]
    assert all(p.expert_template_id in (None, "tpl-maria") for p in prompts)


def test_a_role_with_nobody_on_the_roster_leaves_every_task_with_otto():
    prompts, team = role_tasks.role_plan("HR/People", [], ROSTER[:2])

    assert team == []
    assert all(p.expert_template_id is None for p in prompts)


@pytest.mark.parametrize("user_role", ["Other", "Head of sticker logistics", None])
def test_other_and_unknown_roles_take_the_other_set(user_role: str | None):
    prompts, _ = role_tasks.role_plan(user_role, ["Finding leads"], ROSTER)

    assert [p.title for p in prompts] == [
        t.title for t in role_task_sets.ROLE_FALLBACK_TASKS["Other"]
    ]
    assert all(p.expert_template_id is None for p in prompts)


@pytest.mark.parametrize("user_role", WIZARD_ROLES)
def test_a_role_set_reads_as_a_default_set_but_not_as_the_generic_starters(
    user_role: str,
):
    prompts = role_tasks.role_prompts(user_role, [MARIA, MAX])

    assert role_tasks.is_default_set(prompts)
    assert not role_tasks.is_generic_set(prompts)


def test_the_generic_starters_read_as_both_a_default_and_the_generic_set():
    prompts = intro.fallback_prompts()

    assert role_tasks.is_default_set(prompts)
    assert role_tasks.is_generic_set(prompts)


def test_a_list_the_greeting_wrote_is_neither_default_nor_generic():
    written = [
        SuggestedPrompt(title="Rewrite your Etsy listings", prompt="."),
        *intro.fallback_prompts()[:2],
    ]

    for prompts in (PROMPTS, written):
        assert not role_tasks.is_default_set(prompts)
        assert not role_tasks.is_generic_set(prompts)


def test_an_empty_list_is_neither_default_nor_generic():
    assert not role_tasks.is_default_set([])
    assert not role_tasks.is_generic_set([])


def test_a_default_set_read_back_from_storage_still_reads_as_one():
    role_rows = [
        {**prompt.model_dump(mode="json"), "id": str(index), "state": "sent"}
        for index, prompt in enumerate(role_tasks.role_prompts("Sales/BD", [MAX]))
    ]
    generic_rows = [
        {**prompt.model_dump(mode="json"), "state": "dismissed"}
        for prompt in intro.fallback_prompts()
    ]

    role_set = suggested_tasks.stored_tasks(role_rows)
    generic_set = suggested_tasks.stored_tasks(generic_rows)

    assert len(role_set) == 5 and len(generic_set) == 5
    assert role_tasks.is_default_set(role_set)
    assert role_tasks.is_default_set(generic_set)
    assert role_tasks.is_generic_set(generic_set)
    assert role_tasks.is_default_set(suggested_tasks.as_tasks(intro.fallback_prompts()))
