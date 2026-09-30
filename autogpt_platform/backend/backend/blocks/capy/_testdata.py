"""Canned Capy payloads shared by the block self-tests and unit tests."""

from ._types import (
    Automation,
    Message,
    MessageReceipt,
    Project,
    ProjectRepo,
    ReviewFinding,
    ReviewRound,
    ReviewStarted,
    Task,
    Thread,
    Usage,
)

TEST_PROJECT = Project(
    id="project_01M3F37SAFHACXKK9X6SER0C9E",
    name="autogpt",
    code="AUTO",
    repos=[
        ProjectRepo(repo_full_name="significant-gravitas/autogpt", base_branch="master")
    ],
    created_at="2026-09-26T14:52:18.136Z",
    updated_at="2026-09-26T14:53:09.084Z",
)

TEST_THREAD = Thread(
    id="jam_01M2KY54H0CZC9S7M4DAQZYN7M",
    project_id=TEST_PROJECT.id,
    title="Upgrade CI to Node 24",
    status="working",
    last_model_id="supergrok/grok-4.5",
    usage=Usage(),
    created_at="2026-09-27T10:00:00.000Z",
    updated_at="2026-09-27T10:00:00.000Z",
)

TEST_IDLE_THREAD = TEST_THREAD.model_copy(update={"status": "idle"})

TEST_MESSAGES = [
    Message(
        id="01M0TR02F9YQKN3BRJ6Z21MC7R",
        source="user",
        text="Upgrade the CI pipeline to Node 24 and open a PR.",
        created_at="2026-09-27T10:00:00.000Z",
    ),
    Message(
        id="01M0TR0HZX08ZAVJMDNR9BP58S",
        source="tool",
        text="Update workflow files",
        created_at="2026-09-27T10:02:00.000Z",
    ),
    Message(
        id="01M0TR1302PECXQWQGXNE6EH10",
        source="assistant",
        text="Opened https://github.com/acme/app/pull/12 with the upgrade.",
        created_at="2026-09-27T10:05:00.000Z",
    ),
]

TEST_RECEIPT = MessageReceipt(id="01M0TR3JJZ5YQ248EEV6JY8D6W", deduped=False)

TEST_TASK = Task(
    id="task_01M0TR3JJZ5YQ248EEV6JY8D6W",
    thread_id=TEST_THREAD.id,
    parent_id=TEST_THREAD.id,
    task_path="1",
    title="Update workflow files",
    status="done",
)

TEST_REVIEW_STARTED = ReviewStarted(
    review_id="rev_01M0TR3JJZ5YQ248EEV6JY8D6W",
    request_id="release-gate-481",
    thread_id=TEST_THREAD.id,
    head_sha="b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0",
    adopted=False,
)

TEST_FINDING = ReviewFinding(
    id="finding_1",
    kind="issue",
    severity="high",
    confidence="confirmed",
    category="bug",
    summary="Null check missing before dereference",
    file="src/app.ts",
    line=42,
)

TEST_REVIEW_ROUND = ReviewRound(
    request_id="release-gate-481",
    review_id="rev_01M0TR3JJZ5YQ248EEV6JY8D6W",
    thread_id=TEST_THREAD.id,
    repo="acme/checkout",
    pr_number=481,
    head_sha=TEST_REVIEW_STARTED.head_sha,
    status="completed",
    findings=[TEST_FINDING],
)

TEST_USAGE_REPORT = {
    "from": "2026-09-01T00:00:00.000Z",
    "to": "2026-09-28T00:00:00.000Z",
    "totals": {
        "llmDollars": 1.5,
        "imageDollars": 0,
        "vmDollars": 0.25,
        "totalDollars": 1.75,
    },
    "tokens": {"inputTokens": 1000, "outputTokens": 200},
    "members": [],
    "models": [],
    "images": [],
}

TEST_AUTOMATION = Automation(
    id="automation_01M3PQ8Z4Y7K2N5R9T1V3X6B8D",
    project_id=TEST_PROJECT.id,
    name="Fix new Sentry errors",
    prompt=(
        "Root-cause this Sentry issue. If it is fixable, open a pull request "
        "with a regression test; otherwise explain why not."
    ),
    triggers=[{"type": "sentry", "event": "any_issue"}],
    thread_mode="new",
    max_runs_per_day=10,
    enabled=True,
    created_at="2026-09-30T09:00:00.000Z",
    updated_at="2026-09-30T09:00:00.000Z",
)
