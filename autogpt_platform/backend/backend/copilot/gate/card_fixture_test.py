"""The approval cards the frontend's stories and tests render are built here,
by the real payload builder from real registry blocks, so they cannot drift.

Regenerate with ``UPDATE_CARD_FIXTURE=1``.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.blocks.ayrshare.post_to_x import PostToXBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.blocks.google.gmail import GmailSendBlock
from backend.blocks.google.sheets import GoogleSheetsUpdateRowBlock
from backend.blocks.http import SendWebRequestBlock
from backend.copilot.gate.review import review_id_for, review_payload
from backend.copilot.gate.subject import block_subject, workflow_subject
from backend.data.graph import GraphModel, NodeModel

FIXTURE = (
    Path(__file__).parents[4]
    / "frontend/src/app/(platform)/copilot/components/ApprovalQueue/__tests__"
    / "realCards.json"
)
_BODY = (
    "Hi Dana,\n\nThe Q3 invoice pack is in the shared Invoices folder. Three of "
    "them are still missing a PO number:\n\n- INV-2041\n- INV-2044\n- INV-2051\n\n"
    "Could you look before Friday?\n\nThanks,\nOtto"
)
_BLOCKS: list[tuple[str, Any, dict[str, Any]]] = [
    (
        "Gmail Send",
        GmailSendBlock(),
        {
            "to": ["dana@acme.com"],
            "cc": ["finance@acme.com"],
            "subject": "Q3 invoices missing PO numbers",
            "body": _BODY,
        },
    ),
    (
        "Google Sheets Update Row",
        GoogleSheetsUpdateRowBlock(),
        {
            "spreadsheet": {
                "id": "1x9QwErTyUiOp",
                "name": "Q3 invoices",
                "mimeType": "application/vnd.google-apps.spreadsheet",
                "url": "https://docs.google.com/spreadsheets/d/1x9QwErTyUiOp",
            },
            "sheet_name": "October",
            "row_index": 7,
            "values": ["INV-2044", "4,210.00", "Paid"],
        },
    ),
    (
        "Execute Code",
        ExecuteCodeBlock(),
        {
            "language": "python",
            "setup_commands": ["pip install pandas"],
            "code": (
                "import pandas as pd\n\n"
                "df = pd.read_csv('invoices.csv')\n"
                "missing = df[df.po_number.isna()]\n"
                "print(missing[['id', 'amount']].to_markdown())\n"
            ),
            "timeout": 60,
        },
    ),
    (
        "Send Web Request",
        SendWebRequestBlock(),
        {
            "url": "https://api.acme.com/v2/invoices/2044/status",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-live-9f2",
            },
            "body": {"status": "paid", "paid_at": "2026-09-24", "lines": [1, 2, 3]},
        },
    ),
    (
        "Post To X",
        PostToXBlock(),
        {
            "post": "Q3 is closed. Thanks to everyone who got their invoices in on time.",
            "media_urls": ["https://cdn.acme.com/q3-chart.png"],
            "alt_text": ["Bar chart of Q3 revenue by month"],
            "shorten_links": True,
        },
    ),
]


def build_cards() -> list[dict[str, Any]]:
    cards = [_block_card(*entry) for entry in _BLOCKS]
    cards.append(_workflow_card())
    return json.loads(json.dumps(cards, default=str))


def test_the_frontend_card_fixture_is_what_the_builder_makes():
    cards = build_cards()
    if os.environ.get("UPDATE_CARD_FIXTURE"):
        FIXTURE.write_text(json.dumps(cards, indent=2, ensure_ascii=False) + "\n")
    assert (
        json.loads(FIXTURE.read_text()) == cards
    ), "the approval-card fixture is stale; rerun with UPDATE_CARD_FIXTURE=1"


def _block_card(story: str, block: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    subject = block_subject(block, inputs)
    args = {"id": block.id, "input": inputs}
    return {
        "story": story,
        "review": _row("run_capability", args, subject),
        "schema": block.input_schema.jsonschema(),
    }


def _workflow_card() -> dict[str, Any]:
    send = GmailSendBlock()
    graph = GraphModel(
        id="g-digest",
        version=1,
        name="Morning digest",
        description="",
        user_id="user-1",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        nodes=[
            NodeModel(
                id="send",
                block_id=send.id,
                input_default={},
                graph_id="g-digest",
                graph_version=1,
            )
        ],
        links=[],
        sub_graphs=[],
    )
    args = {"library_agent_id": "lib-7", "inputs": {"topic": "Q3 invoices"}}
    return {
        "story": "Workflow",
        "review": _row("run_agent", args, workflow_subject(graph)),
        "schema": None,
    }


def _row(tool: str, args: dict[str, Any], subject: Any) -> dict[str, Any]:
    review_id = review_id_for("session-1", "user-1", tool, args)
    node_id = review_id.split(":")[0]
    return {
        "node_exec_id": review_id,
        "node_id": node_id,
        "user_id": "user-1",
        "session_id": "s1",
        "graph_exec_id": None,
        "graph_id": None,
        "graph_version": None,
        "payload": review_payload(
            tool,
            args,
            subject,
            reason=subject.reason,
            reason_kind="subject" if subject.reason else "mode",
            mode="auto",
            tool_call_id=f"call-{tool}",
            turn=1,
        ),
        "instructions": subject.name,
        "editable": False,
        "status": "WAITING",
    }
