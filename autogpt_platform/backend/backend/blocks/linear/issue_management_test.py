import importlib
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.blocks.linear._api import LinearClient
from backend.blocks.linear._config import (
    TEST_CREDENTIALS_INPUT_OAUTH,
    TEST_CREDENTIALS_OAUTH,
)
from backend.blocks.linear.issues import LinearCreateIssueBlock

ISSUE_ID = "1108b684-5175-4f44-95b5-f1e9cd95f821"
STATE_ID = "949f1c34-21ce-48b7-b4b7-88a013c24257"
USER_ID = "2e41b861-8642-4898-94b1-d0093c406498"
LABEL_ID = "37f36e09-3fb8-42e4-b615-f7b53c61d740"
OTHER_LABEL_ID = "c5660b9d-6a6a-47e3-a0f0-b8c6a0d405c0"
ISSUE = {
    "id": ISSUE_ID,
    "identifier": "ENG-123",
    "title": "Updated title",
    "description": "",
    "priority": 0,
}


@pytest.fixture
def update_module():
    return importlib.import_module("backend.blocks.linear.issue_update")


@pytest.fixture
def lifecycle_module():
    return importlib.import_module("backend.blocks.linear.issue_lifecycle")


async def test_create_forwards_parent_and_zero_priority(monkeypatch):
    monkeypatch.setattr(
        LinearClient, "try_get_team_by_name", AsyncMock(return_value="team")
    )
    mutate = AsyncMock(return_value={"issueCreate": {"issue": ISSUE}})
    monkeypatch.setattr(LinearClient, "mutate", mutate)
    block = LinearCreateIssueBlock()
    data = block.Input(
        credentials=TEST_CREDENTIALS_INPUT_OAUTH,
        title="Child",
        description="",
        team_name="Engineering",
        parent_id="ENG-122",
        priority=0,
    )
    outputs = dict(
        [item async for item in block.run(data, credentials=TEST_CREDENTIALS_OAUTH)]
    )
    assert outputs["issue_id"] == "ENG-123"
    assert mutate.call_args.args[1]["input"] == {
        "teamId": "team",
        "title": "Child",
        "parentId": "ENG-122",
        "description": "",
        "priority": 0,
    }


async def test_update_only_requested_fields_and_preserves_falsey_values(
    update_module, monkeypatch
):
    mutate = AsyncMock(return_value={"issueUpdate": {"success": True, "issue": ISSUE}})
    monkeypatch.setattr(LinearClient, "mutate", mutate)
    block = update_module.LinearUpdateIssueBlock()
    data = block.Input(
        credentials=TEST_CREDENTIALS_INPUT_OAUTH,
        issue_id="ENG-123",
        changes={"priority": 0, "description": "", "label_ids": []},
    )
    outputs = dict(
        [item async for item in block.run(data, credentials=TEST_CREDENTIALS_OAUTH)]
    )
    assert outputs["issue"].id == ISSUE_ID
    assert mutate.call_args.args[1] == {
        "id": "ENG-123",
        "input": {"priority": 0, "description": "", "labelIds": []},
    }
    assert "IssueUpdateInput!" in mutate.call_args.args[0]


async def test_update_state_assignee_date_estimate_and_atomic_labels(
    update_module, monkeypatch
):
    mutate = AsyncMock(return_value={"issueUpdate": {"success": True, "issue": ISSUE}})
    monkeypatch.setattr(LinearClient, "mutate", mutate)
    query = AsyncMock(side_effect=AssertionError("No read-modify-write for labels"))
    monkeypatch.setattr(LinearClient, "query", query)
    block = update_module.LinearUpdateIssueBlock()
    data = block.Input(
        credentials=TEST_CREDENTIALS_INPUT_OAUTH,
        issue_id=ISSUE_ID,
        changes={
            "title": "New title",
            "state_id": STATE_ID,
            "assignee_id": USER_ID,
            "due_date": "2026-10-01",
            "estimate": 0,
            "add_label_ids": [LABEL_ID],
            "remove_label_ids": [OTHER_LABEL_ID],
        },
    )
    _ = [item async for item in block.run(data, credentials=TEST_CREDENTIALS_OAUTH)]
    assert mutate.call_args.args[1]["input"] == {
        "title": "New title",
        "stateId": STATE_ID,
        "assigneeId": USER_ID,
        "dueDate": "2026-10-01",
        "estimate": 0,
        "addedLabelIds": [LABEL_ID],
        "removedLabelIds": [OTHER_LABEL_ID],
    }
    query.assert_not_called()


async def test_explicit_clear_fields_send_null(update_module, monkeypatch):
    mutate = AsyncMock(return_value={"issueUpdate": {"success": True, "issue": ISSUE}})
    monkeypatch.setattr(LinearClient, "mutate", mutate)
    block = update_module.LinearUpdateIssueBlock()
    data = block.Input(
        credentials=TEST_CREDENTIALS_INPUT_OAUTH,
        issue_id=ISSUE_ID,
        changes={"clear_fields": ["assignee", "description", "due_date", "estimate"]},
    )
    _ = [item async for item in block.run(data, credentials=TEST_CREDENTIALS_OAUTH)]
    assert mutate.call_args.args[1]["input"] == {
        "assigneeId": None,
        "description": None,
        "dueDate": None,
        "estimate": None,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"title": None},
        {"add_label_ids": []},
        {"title": " "},
        {"state_id": "Done"},
        {"assignee_id": "Alice"},
        {"estimate": -1},
        {"estimate": 1.5},
        {"priority": 5},
        {"due_date": "2026-02-30"},
        {"label_ids": [], "add_label_ids": [LABEL_ID]},
        {"add_label_ids": [LABEL_ID], "remove_label_ids": [LABEL_ID]},
        {"assignee_id": USER_ID, "clear_fields": ["assignee"]},
        {"description": "", "clear_fields": ["description"]},
        {"clear_fields": ["status"]},
        {"titel": "Typo"},
    ],
)
def test_invalid_or_ambiguous_updates_rejected_before_request(update_module, changes):
    with pytest.raises(ValidationError):
        update_module.LinearUpdateIssueBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT_OAUTH,
            issue_id=ISSUE_ID,
            changes=changes,
        )


@pytest.mark.parametrize(
    "response",
    [
        {"success": False, "issue": ISSUE},
        {"success": True, "issue": None},
    ],
)
async def test_unsuccessful_update_does_not_emit_success(
    update_module, monkeypatch, response
):
    monkeypatch.setattr(
        LinearClient, "mutate", AsyncMock(return_value={"issueUpdate": response})
    )
    block = update_module.LinearUpdateIssueBlock()
    data = block.Input(
        credentials=TEST_CREDENTIALS_INPUT_OAUTH,
        issue_id=ISSUE_ID,
        changes={"priority": 1},
    )
    with pytest.raises(ValueError, match="update"):
        _ = [item async for item in block.run(data, credentials=TEST_CREDENTIALS_OAUTH)]


@pytest.mark.parametrize(
    "class_name,operation",
    [
        ("LinearArchiveIssueBlock", "issueArchive"),
        ("LinearDeleteIssueBlock", "issueDelete"),
    ],
)
async def test_lifecycle_mutations_have_distinct_semantics(
    lifecycle_module, monkeypatch, class_name, operation
):
    mutate = AsyncMock(return_value={operation: {"success": True}})
    monkeypatch.setattr(LinearClient, "mutate", mutate)
    block = vars(lifecycle_module)[class_name]()
    data = block.Input(credentials=TEST_CREDENTIALS_INPUT_OAUTH, issue_id="ENG-123")
    assert dict(
        [item async for item in block.run(data, credentials=TEST_CREDENTIALS_OAUTH)]
    ) == {"issue_id": "ENG-123", "success": True}
    assert operation + "(" in mutate.call_args.args[0]
    assert mutate.call_args.args[1] == {"id": "ENG-123"}
    assert "permanentlyDelete" not in mutate.call_args.args[0]


@pytest.mark.parametrize(
    "class_name,operation",
    [
        ("LinearArchiveIssueBlock", "issueArchive"),
        ("LinearDeleteIssueBlock", "issueDelete"),
    ],
)
async def test_unsuccessful_lifecycle_operation_raises(
    lifecycle_module, monkeypatch, class_name, operation
):
    monkeypatch.setattr(
        LinearClient, "mutate", AsyncMock(return_value={operation: {"success": False}})
    )
    block = vars(lifecycle_module)[class_name]()
    data = block.Input(credentials=TEST_CREDENTIALS_INPUT_OAUTH, issue_id=ISSUE_ID)
    with pytest.raises(ValueError):
        _ = [item async for item in block.run(data, credentials=TEST_CREDENTIALS_OAUTH)]


def test_new_mutations_require_write_scope_and_review_gate(
    update_module, lifecycle_module
):
    for cls in (
        update_module.LinearUpdateIssueBlock,
        lifecycle_module.LinearArchiveIssueBlock,
        lifecycle_module.LinearDeleteIssueBlock,
    ):
        block = cls()
        assert block.is_sensitive_action
        info = block.input_schema.get_credentials_fields_info()["credentials"]
        assert "write" in info.required_scopes
