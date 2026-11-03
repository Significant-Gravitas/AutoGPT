import asyncio
import sys
import uuid
from datetime import date, datetime, timedelta, timezone

import prisma.models

from backend.api.features.tasks import db
from backend.api.features.tasks.models import (
    ActivityKind,
    Actor,
    AssigneeRef,
    AssigneeType,
    AttachmentCreate,
    TaskCreate,
    TaskPriority,
    TaskStatus,
)
from backend.data import db as prisma_db

DEMO_LABEL = "demo"
EXPERTS = [
    ("Maria", "Marketing", "sky-300"),
    ("Leo", "Research", "amber-300"),
    ("Ada", "Engineering", "emerald-300"),
]


async def ensure_experts(user_id: str) -> dict[str, prisma.models.Expert]:
    result: dict[str, prisma.models.Expert] = {}
    for name, role, color in EXPERTS:
        expert = await prisma.models.Expert.prisma().find_first(
            where={"ownerUserId": user_id, "name": name, "isArchived": False}
        )
        if expert is None:
            expert = await prisma.models.Expert.prisma().create(
                data={
                    "ownerUserId": user_id,
                    "name": name,
                    "role": role,
                    "color": color,
                    "identity": f"{name} handles {role.lower()} work for the team.",
                    "tagline": f"Your {role.lower()} expert",
                }
            )
        result[name] = expert
    return result


def _assign(expert: prisma.models.Expert) -> AssigneeRef:
    return AssigneeRef(type=AssigneeType.EXPERT, id=expert.id)


def _actor(expert: prisma.models.Expert) -> Actor:
    return Actor.expert(expert.id, expert.name, expert.avatarUrl, expert.color or None)


async def add_turn(
    user_id: str,
    task_id: str,
    expert: prisma.models.Expert,
    turn_index: int,
    summary: str,
    cards: list[dict],
    tokens: int,
    cost_usd: float,
    duration_ms: int,
) -> None:
    input_tokens = int(tokens * 0.82)
    await db.add_activity(
        user_id,
        task_id,
        ActivityKind.TURN,
        _actor(expert),
        data={
            "session_id": str(uuid.uuid4()),
            "turn_index": turn_index,
            "status": "completed",
            "tool_calls": len(cards),
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": tokens - input_tokens,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "summary": summary,
            "tool_call_cards": cards,
        },
    )
    await db.add_turn_metrics(task_id, len(cards), tokens, cost_usd, duration_ms)


def card(name: str, title: str, input_preview: str, output_preview: str) -> dict:
    return {
        "id": f"toolu_{uuid.uuid4().hex[:20]}",
        "name": name,
        "title": title,
        "status": "completed",
        "input_preview": input_preview,
        "output_preview": output_preview,
    }


async def seed(email: str) -> None:
    user = await prisma.models.User.prisma().find_unique(where={"email": email})
    if user is None:
        raise SystemExit(f"No user with email {email}")
    user_id = user.id
    existing = await prisma.models.Task.prisma().count(
        where={"userId": user_id, "labels": {"has": DEMO_LABEL}}
    )
    if existing:
        print(f"{email} already has {existing} demo tasks; skipping")
        return
    experts = await ensure_experts(user_id)
    await seed_tasks(user_id, experts)
    print(f"Seeded demo tasks for {email}")


async def create(user_id: str, data: TaskCreate, actor: Actor | None = None):
    data.labels = [*(data.labels or []), DEMO_LABEL]
    return await db.create_task(user_id, data, actor or Actor.user(user_id))


async def seed_tasks(user_id: str, experts: dict[str, prisma.models.Expert]) -> None:
    maria, leo, ada = experts["Maria"], experts["Leo"], experts["Ada"]
    me = Actor.user(user_id)
    today = date.today()

    launch = await create(
        user_id,
        TaskCreate(
            title="Launch the November newsletter",
            description="Ship the November issue to the full list by Friday.",
            status=TaskStatus.IN_PROGRESS,
            priority=TaskPriority.HIGH,
            assignee=AssigneeRef(type=AssigneeType.OTTO),
            due_date=today + timedelta(days=3),
            labels=["launch"],
            budget_cap_usd=5.0,
        ),
        Actor.otto(),
    )
    research = await create(
        user_id,
        TaskCreate(
            title="Research what subscribers clicked last month",
            parent_id=launch.id,
            assignee=_assign(leo),
            priority=TaskPriority.MEDIUM,
        ),
        Actor.otto(),
    )
    await db.set_task_status(
        user_id,
        research.id,
        TaskStatus.DONE,
        _actor(leo),
        outcome_summary=(
            "Top clicks: the pricing explainer (31%), the workflow templates "
            "roundup (24%) and the customer story (18%)."
        ),
    )
    await add_turn(
        user_id,
        research.id,
        leo,
        1,
        "Pulled October click data and ranked the links",
        [
            card(
                "web_search",
                "Web search",
                "newsletter benchmarks 2026",
                "8 results",
            ),
            card(
                "read_workspace_file",
                "Read file",
                "october-clicks.csv",
                "412 rows",
            ),
            card("run_python", "Run code", "rank links by CTR", "Top 3 links"),
        ],
        tokens=18420,
        cost_usd=0.05,
        duration_ms=94000,
    )
    await db.add_attachment(
        user_id,
        research.id,
        AttachmentCreate(kind="file", title="october-click-report.md"),
        _actor(leo),
    )

    draft = await create(
        user_id,
        TaskCreate(
            title="Draft the newsletter copy",
            parent_id=launch.id,
            assignee=_assign(maria),
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            due_date=today + timedelta(days=1),
        ),
        Actor.otto(),
    )
    await add_turn(
        user_id,
        draft.id,
        maria,
        1,
        "Drafted three subject lines and the intro",
        [
            card(
                "read_workspace_file",
                "Read file",
                "october-click-report.md",
                "Top clicks summary",
            ),
            card("web_search", "Web search", "subject line length", "6 results"),
            card(
                "write_workspace_file",
                "Write file",
                "november-draft.md",
                "Saved 1.2 KB",
            ),
            card("web_fetch", "Read page", "brand voice guide", "Voice notes"),
        ],
        tokens=12034,
        cost_usd=0.04,
        duration_ms=80000,
    )
    await db.request_input(
        user_id,
        draft.id,
        "Which tone should the intro use?",
        ["Formal", "Casual", "Playful"],
        _actor(maria),
    )

    await create(
        user_id,
        TaskCreate(
            title="Schedule the send in the email tool",
            parent_id=launch.id,
            assignee=_assign(ada),
            priority=TaskPriority.MEDIUM,
            due_date=today + timedelta(days=3),
        ),
        Actor.otto(),
    )

    pricing = await create(
        user_id,
        TaskCreate(
            title="Fix the broken pricing page link",
            assignee=_assign(ada),
            priority=TaskPriority.URGENT,
            status=TaskStatus.IN_PROGRESS,
            labels=["website"],
        ),
    )
    await db.set_live(
        pricing.id,
        True,
        "Checking the site for other broken links",
        datetime.now(timezone.utc) - timedelta(seconds=40),
    )
    await add_turn(
        user_id,
        pricing.id,
        ada,
        1,
        "Found the redirect that drops the plan parameter",
        [
            card("web_fetch", "Read page", "/pricing", "301 to /plans"),
            card("run_python", "Run code", "crawl internal links", "2 broken"),
        ],
        tokens=9310,
        cost_usd=0.03,
        duration_ms=51000,
    )
    await db.add_comment(
        user_id,
        pricing.id,
        "@Ada the footer link has the same problem, please fix both.",
        me,
    )

    await create(
        user_id,
        TaskCreate(
            title="Write three social posts for the launch",
            assignee=_assign(maria),
            priority=TaskPriority.MEDIUM,
            due_date=today + timedelta(days=4),
            labels=["launch", "social"],
        ),
    )

    competitors = await create(
        user_id,
        TaskCreate(
            title="Summarise competitor pricing changes",
            assignee=_assign(leo),
            priority=TaskPriority.LOW,
            status=TaskStatus.IN_PROGRESS,
        ),
    )
    await db.set_task_status(
        user_id,
        competitors.id,
        TaskStatus.DONE,
        _actor(leo),
        outcome_summary=(
            "Two of five competitors raised prices by 10-15%; one added a free "
            "tier. Full table attached."
        ),
    )
    await db.add_attachment(
        user_id,
        competitors.id,
        AttachmentCreate(kind="text", title="competitor-pricing.md"),
        _actor(leo),
    )
    await db.add_comment(user_id, competitors.id, "Thanks @Leo, very useful.", me)

    await create(
        user_id,
        TaskCreate(
            title="Review the onboarding workflow",
            assignee=AssigneeRef(type=AssigneeType.USER),
            priority=TaskPriority.MEDIUM,
            due_date=today - timedelta(days=1),
        ),
    )
    await create(
        user_id,
        TaskCreate(
            title="Plan the Q1 webinar series",
            status=TaskStatus.BACKLOG,
            priority=TaskPriority.NONE,
            labels=["events"],
        ),
    )


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m backend.api.features.tasks.seed_demo <user_email>"
        )
    await prisma_db.connect()
    try:
        await seed(sys.argv[1])
    finally:
        await prisma_db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
