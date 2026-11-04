"""The account every rubric example runs in.

The examples were written for a working business: "this CSV" was attached,
"the editor expert" is hired, "the last run" exists. An empty account turns
half of them into a reasonable question about something missing, which
measures the fixture instead of the rubric. So each turn starts with the
first-turn context production prepends (business profile, hired team,
standing work), what the user attached or pasted, and tools that answer from
this account. When Otto still asks, the user answers once and the turn goes
on, so the action it finally takes is what gets scored.
"""

import base64
import json
from datetime import UTC, datetime
from typing import Any

from backend.api.features.experts.models import PROTECTED_SOUL_RULES, Expert
from backend.copilot.eval.style.assembly import user_prefix
from backend.copilot.service import (
    _sanitize_user_context_field,
    format_user_context_prefix,
)
from backend.copilot.tools.models import (
    AgentInfo,
    AgentOutputResponse,
    AgentsFoundResponse,
    ExecutionOutputInfo,
    MemorySearchResponse,
    TaskListToolResponse,
    WorkspaceFileInfoData,
    WorkspaceFolderInfoData,
)
from backend.copilot.tools.workspace_files import (
    WorkspaceFileContentResponse,
    WorkspaceFileListResponse,
)
from backend.data.understanding import (
    BusinessUnderstanding,
    format_understanding_for_prompt,
)

BUSINESS = BusinessUnderstanding(
    id="task-rubric-eval",
    user_id="task-rubric-eval",
    created_at=datetime(2026, 1, 1, tzinfo=UTC),
    updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    user_name="Priya",
    job_title="Founder and CEO",
    business_name="Brightwave",
    industry="B2B SaaS: a team scheduling app",
    business_size="12 people",
    key_workflows=["Inbound leads", "Customer support", "Monthly board reporting"],
    current_software=[
        "Stripe (billing and signups)",
        "HubSpot (CRM)",
        "Typeform (lead form on the website)",
        "Gmail (support@brightwave.io inbox)",
        "Linear (engineering)",
        "GitHub (brightwave/app)",
        "QuickBooks (accounting)",
        "WordPress (blog)",
        "Intercom (help center at help.brightwave.io)",
        "Google Sheets",
    ],
    additional_notes=(
        "Based in Bengaluru. Website brightwave.io. Competitors: Calendly, "
        "Cal.com, SavvyCal, Reclaim. Sales team: sales@brightwave.io. Q1 "
        "launch is Brightwave Teams 2.0 on 15 Jan. Opening a Shopify merch "
        "store (t-shirts, mugs, stickers) for the user community."
    ),
)


def _expert(slug: str, name: str, role: str) -> Expert:
    return Expert(
        id=f"expert-{slug}",
        name=name,
        avatar_url=None,
        role=role,
        tagline=None,
        bio=None,
        skills=[],
        identity=f"{name}, {role}.",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )


TEAM = [
    _expert("maya", "Maya", "Editor"),
    _expert("leo", "Leo", "Designer"),
    _expert("rex", "Rex", "Lead enrichment specialist"),
]

# The open tasks the examples name ("On T-12…", "Retry T-15", "the onboarding
# task"), with the detail a real board shows, so "update the open task" is a
# choice the model can make without asking what the task was about.
BOARD = [
    ("T-4", "in_review", "Write the welcome email sequence", "Maya"),
    ("T-7", "in_progress", "Research competitor pricing", "Otto"),
    ("T-9", "done", "Launch the referral campaign in the US", "Otto"),
    ("T-12", "in_review", "Draft the launch blog post", "Maya"),
    (
        "T-15",
        "blocked",
        (
            "Publish the Teams 2.0 changelog to the blog (the last attempt "
            "failed: WordPress returned 401, reconnected since)"
        ),
        "Otto",
    ),
    (
        "T-16",
        "blocked",
        "Onboarding plan (waiting on you: option A or B)",
        "Otto",
    ),
    ("T-18", "in_progress", "Landing page", "Leo"),
]

FILES = [
    ("file-customers", "customers.csv", "text/csv", 48_200),
    ("file-leads", "leads-2000.csv", "text/csv", 310_000),
]
INTERVIEWS = [f"file-interview-{i:02}" for i in range(1, 41)]

WORKFLOWS = [
    ("lib-invoice-sync", "graph-invoice-sync", "Invoice sync to QuickBooks"),
    ("lib-lead-enrichment", "graph-lead-enrichment", "Lead enrichment"),
]

# What the user pasted or attached with the message, keyed by example id.
PASTED = {
    26: (
        "\n\nFrom: Dan Okafor <dan@northwind.io>\nSubject: Renewal\n\nHi "
        "Priya, our plan renews next week. Can we move to annual billing and "
        "add 5 seats? Thanks, Dan"
    ),
    39: "\n\nananya.rao@northwind.io",
    40: (
        "\n\nBrightwave finds a meeting time for your whole team in seconds, "
        "across every calendar and time zone."
    ),
}
ATTACHED = {
    23: INTERVIEWS,
    24: ["file-customers"],
    38: ["file-leads"],
}

DEFAULT_ANSWER = (
    "I don't have more to add. Use what you know about Brightwave and "
    "sensible defaults, and go ahead."
)
# Where a default answer can't stand in for the user: the examples whose
# point is that Otto must ask, and the one that names no competitor.
ANSWERS = {
    11: "Calendly. Their pricing page is calendly.com/pricing.",
    16: "More trial signups this quarter. We get about 400 a month now.",
    17: "Fewer customers cancelling in their first month.",
    36: "Our new AI meeting-notes add-on. A .com under $20 a year.",
}


def first_message(example_id: int, text: str) -> str:
    """The user's message as production sends it on a first turn: the
    business profile, the team and standing work, then what they typed,
    pasted and attached."""
    understanding = _sanitize_user_context_field(
        format_understanding_for_prompt(BUSINESS)
    )
    message = text + PASTED.get(example_id, "")
    if files := ATTACHED.get(example_id):
        message += (
            "\n\n[Attached files]\n"
            + "\n".join(f"- file_id={fid}" for fid in files)
            + "\nUse read_workspace_file with the file_id to access file contents."
        )
    return format_user_context_prefix(understanding) + user_prefix(None, TEAM) + message


def answer(example_id: int) -> str:
    return ANSWERS.get(example_id, DEFAULT_ANSWER)


def account_result(name: str, args: dict[str, Any]) -> str | None:
    """This account's answer to a lookup, or None for the shared stub."""
    if name == "list_my_tasks":
        return _board()
    if name == "list_workspace_files":
        return _files()
    if name == "read_workspace_file":
        return _file(str(args.get("file_id") or args.get("path") or ""))
    if name in ("find_library_agent", "find_agent"):
        return _library()
    if name == "view_agent_output":
        return _last_run()
    if name == "memory_search":
        return MemorySearchResponse(
            message="Found 2 memories.",
            facts=[
                "Brightwave's tone of voice is friendly and plain, never formal.",
                "Priya wants drafts reviewed by Maya before anything is sent.",
            ],
            recent_episodes=[],
        ).model_dump_json()
    return None


def _board() -> str:
    return TaskListToolResponse(
        message="\n".join(
            f"{key} [{status}] {title}, assigned to {owner} (id task-{key[2:]})"
            for key, status, title, owner in BOARD
        ),
        tasks=[
            {"key": key, "status": status, "title": title, "assignee": owner}
            for key, status, title, owner in BOARD
        ],
    ).model_dump_json()


def _files() -> str:
    files = [
        WorkspaceFileInfoData(
            file_id=fid, name=name, path=f"/{name}", mime_type=mime, size_bytes=size
        )
        for fid, name, mime, size in FILES
    ]
    return WorkspaceFileListResponse(
        message=f"{len(files)} files and 1 folder.",
        files=files,
        total_count=len(files),
        folders=[
            WorkspaceFolderInfoData(
                folder_id="folder-interviews",
                name="customer-interviews",
                file_count=len(INTERVIEWS),
            )
        ],
    ).model_dump_json()


FILE_CONTENT = {
    "file-customers": (
        "name,email,company,plan\nDan Okafor,dan@northwind.io,Northwind,Team\n"
        "Dan Okafor,dan@northwind.io,Northwind,Team\nLi Wei,li@acme.co,Acme,Pro\n"
        "... 1,204 rows in total, 87 of them duplicates\n"
    ),
    "file-leads": (
        "email,first_name,company\nananya.rao@northwind.io,Ananya,Northwind\n"
        "tom@globex.com,Tom,Globex\n... 2,000 rows in total\n"
    ),
}
INTERVIEW_CONTENT = (
    "Interviewer: What made you switch?\nCustomer: Scheduling across time "
    "zones took us an hour a week. ... (a 40-minute transcript)"
)


def _file(file_id: str) -> str:
    content = FILE_CONTENT.get(file_id, INTERVIEW_CONTENT)
    return WorkspaceFileContentResponse(
        message=f"Contents of {file_id}.",
        file_id=file_id,
        name=file_id,
        path=f"/{file_id}",
        mime_type="text/plain",
        content_base64=base64.b64encode(content.encode()).decode(),
    ).model_dump_json()


def _library() -> str:
    return AgentsFoundResponse(
        message=f"Found {len(WORKFLOWS)} agents in your library.",
        agents=[
            AgentInfo(
                id=library_id,
                name=name,
                description=name,
                source="library",
                in_library=True,
                graph_id=graph_id,
            )
            for library_id, graph_id, name in WORKFLOWS
        ],
        count=len(WORKFLOWS),
    ).model_dump_json()


def _last_run() -> str:
    return AgentOutputResponse(
        message="The last run of Invoice sync to QuickBooks failed.",
        agent_name="Invoice sync to QuickBooks",
        agent_id="graph-invoice-sync",
        library_agent_id="lib-invoice-sync",
        execution=ExecutionOutputInfo(
            execution_id="exec-invoice-sync-last",
            status="FAILED",
            outputs={
                "error": [
                    json.dumps(
                        {
                            "node": "QuickBooks",
                            "error": "401 Unauthorized: token expired",
                        }
                    )
                ]
            },
        ),
        total_executions=12,
    ).model_dump_json()
