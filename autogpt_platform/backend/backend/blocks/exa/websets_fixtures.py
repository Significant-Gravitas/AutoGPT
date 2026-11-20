"""
Spec-shaped Websets API responses for block self-tests and unit tests.

Each builder returns a fresh dict so a test can mutate its copy freely.
"""

from typing import Any

from backend.sdk import Block

from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT

WEBSET_ID = "webset_test123"
SEARCH_ID = "search_test123"
MONITOR_ID = "monitor_test123"
RUN_ID = "run_test123"
IMPORT_ID = "import_test123"
TIMESTAMP = "2024-01-01T00:00:00Z"


def sample_search(**overrides: Any) -> dict[str, Any]:
    search = {
        "id": SEARCH_ID,
        "object": "webset_search",
        "websetId": WEBSET_ID,
        "status": "completed",
        "query": "AI startups in Europe",
        "entity": {"type": "company"},
        "criteria": [{"description": "Raised a Series A", "successRate": 80}],
        "count": 10,
        "maxPeoplePerCompany": None,
        "behavior": "override",
        "exclude": [],
        "scope": [],
        "progress": {"found": 5, "analyzed": 20, "completion": 100, "timeLeft": 0},
        "recall": None,
        "metadata": {},
        "canceledAt": None,
        "canceledReason": None,
        "createdAt": TIMESTAMP,
        "updatedAt": TIMESTAMP,
    }
    search.update(overrides)
    return search


def sample_webset(**overrides: Any) -> dict[str, Any]:
    webset = {
        "id": WEBSET_ID,
        "object": "webset",
        "status": "idle",
        "externalId": None,
        "title": "AI startups",
        "searches": [sample_search()],
        "imports": [],
        "enrichments": [],
        "monitors": [],
        "excludes": [],
        "metadata": {},
        "dashboardUrl": f"https://websets.exa.ai/{WEBSET_ID}",
        "createdAt": TIMESTAMP,
        "updatedAt": TIMESTAMP,
    }
    webset.update(overrides)
    return webset


def sample_preview(**overrides: Any) -> dict[str, Any]:
    preview = {
        "search": {
            "entity": {"type": "company"},
            "criteria": [{"description": "Based in Europe"}],
        },
        "enrichments": [{"description": "Company website", "format": "url"}],
        "items": [],
    }
    preview.update(overrides)
    return preview


def sample_monitor_run(**overrides: Any) -> dict[str, Any]:
    run = {
        "id": RUN_ID,
        "object": "monitor_run",
        "monitorId": MONITOR_ID,
        "status": "completed",
        "type": "search",
        "completedAt": TIMESTAMP,
        "failedAt": None,
        "failedReason": None,
        "canceledAt": None,
        "createdAt": TIMESTAMP,
        "updatedAt": TIMESTAMP,
    }
    run.update(overrides)
    return run


def sample_monitor(**overrides: Any) -> dict[str, Any]:
    monitor = {
        "id": MONITOR_ID,
        "object": "monitor",
        "status": "enabled",
        "websetId": WEBSET_ID,
        "cadence": {"cron": "0 9 * * 1", "timezone": "Etc/UTC"},
        "behavior": {
            "type": "search",
            "config": {"query": "AI startups", "count": 10, "behavior": "append"},
        },
        "lastRun": None,
        "nextRunAt": TIMESTAMP,
        "metadata": {},
        "createdAt": TIMESTAMP,
        "updatedAt": TIMESTAMP,
    }
    monitor.update(overrides)
    return monitor


def sample_import(**overrides: Any) -> dict[str, Any]:
    imported = {
        "id": IMPORT_ID,
        "object": "import",
        "status": "completed",
        "format": "csv",
        "entity": {"type": "company"},
        "title": "Customer list",
        "count": 2,
        "metadata": {},
        "failedReason": None,
        "failedAt": None,
        "failedMessage": None,
        "createdAt": TIMESTAMP,
        "updatedAt": TIMESTAMP,
    }
    imported.update(overrides)
    return imported


def sample_team(**overrides: Any) -> dict[str, Any]:
    team = {
        "object": "team",
        "id": "team_test123",
        "name": "Test team",
        "concurrency": {"active": 2, "queued": 0},
        "limits": {"maxConcurrent": 10, "maxQueued": None},
    }
    team.update(overrides)
    return team


async def collect_outputs(block: Block, **inputs: Any) -> list[tuple[str, Any]]:
    """Run ``block`` with test credentials and return everything it yields."""
    input_data = block.input_schema.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **inputs}
    )
    return [
        output async for output in block.run(input_data, credentials=TEST_CREDENTIALS)
    ]
