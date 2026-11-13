"""render_ui validates the cards Otto shows and decides who can be offered it."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.permissions import CopilotPermissions, all_known_tool_names
from backend.copilot.tools.models import ErrorResponse, ResponseType
from backend.copilot.tools.render_ui import (
    RENDER_UI_TOOL,
    RenderUITool,
    render_ui_disabled_tools,
    shows_cards,
    without_render_ui,
)
from backend.copilot.tools.render_ui_models import (
    MAX_BLOCKS,
    MAX_PAYLOAD_CHARS,
    UIBlocksResponse,
    render_ui_parameters,
)

FIXTURE = (
    Path(__file__).parents[4]
    / "frontend/src/app/(platform)/copilot/tools/RenderUI/__tests__"
    / "renderUIOutput.json"
)

ALL_BLOCKS: list[dict[str, Any]] = [
    {
        "type": "stats",
        "items": [
            {"label": "Leads", "value": "412", "change": "+6%", "tone": "positive"},
            {"label": "Replies", "value": "96", "change": "+37%", "tone": "positive"},
        ],
    },
    {
        "type": "chart",
        "title": "Replies per week",
        "kind": "area",
        "x_labels": ["Wk 1", "Wk 2", "Wk 3", "Wk 4"],
        "series": [{"name": "This month", "values": [14, 17, 28, 37]}],
        "annotation": {"x": "Wk 3", "label": "New subject lines"},
    },
    {
        "type": "workflow_diagram",
        "title": "Where leads drop out",
        "steps": [
            {"label": "New lead", "detail": "412", "status": "ok"},
            {"label": "Enrich lead", "detail": "31 failed", "status": "error"},
        ],
    },
    {
        "type": "gallery",
        "images": [
            {"src": "workspace://file-1#image/png", "caption": "Before"},
            {"src": "workspace://file-2", "caption": "After"},
        ],
    },
    {
        "type": "timeline",
        "items": [{"date": "14 Oct", "title": "New subject lines", "status": "ok"}],
    },
    {
        "type": "options",
        "items": [
            {"title": "Fix enrichment", "description": "Recover 31 leads"},
            {"title": "Add a day 5 follow-up", "tone": "positive"},
        ],
    },
    {
        "type": "table",
        "columns": ["Metric", "This month"],
        "rows": [["Meetings", "21"]],
    },
]


def block(block_type: str) -> dict[str, Any]:
    return next(item for item in ALL_BLOCKS if item["type"] == block_type)


@pytest.fixture()
def tool() -> RenderUITool:
    return RenderUITool()


@pytest.fixture()
def session() -> ChatSession:
    return ChatSession.new(
        user_id="test-user", dry_run=False, session_id="fixture-session"
    )


async def run(tool: RenderUITool, session: ChatSession, blocks: Any):
    return await tool._execute(user_id=None, session=session, blocks=blocks)


@pytest.mark.asyncio
@pytest.mark.parametrize("sample", ALL_BLOCKS, ids=lambda item: item["type"])
async def test_every_block_type_is_returned(
    tool: RenderUITool, session: ChatSession, sample: dict[str, Any]
):
    result = await run(tool, session, [sample])

    assert isinstance(result, UIBlocksResponse)
    assert result.type == ResponseType.UI_BLOCKS
    assert result.blocks[0].type == sample["type"]


@pytest.mark.asyncio
async def test_output_matches_frontend_fixture(
    tool: RenderUITool, session: ChatSession
):
    """The frontend renders this same file, so the two halves can't drift."""
    result = await run(tool, session, ALL_BLOCKS[:MAX_BLOCKS])
    assert isinstance(result, UIBlocksResponse)
    output = json.loads(result.model_dump_json(exclude_none=True))

    if os.environ.get("UPDATE_RENDER_UI_FIXTURE"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(output, indent=2) + "\n")

    assert output == json.loads(FIXTURE.read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_blocks, expected",
    [
        pytest.param([], "blocks", id="no_blocks"),
        pytest.param([block("table")] * 7, "at most 6", id="seven_blocks"),
        pytest.param(
            [{**block("options"), "items": [{"title": "Go", "href": "https://x.io"}]}],
            "href",
            id="unknown_field",
        ),
        pytest.param(
            [{"type": "html", "markup": "<b>hi</b>"}], "blocks", id="unknown_type"
        ),
        pytest.param(
            [
                {
                    **block("chart"),
                    "series": [{"name": "Short", "values": [1, 2]}],
                }
            ],
            "x_labels",
            id="chart_series_must_match_labels",
        ),
        pytest.param(
            [{**block("chart"), "annotation": {"x": "Wk 9", "label": "Launch"}}],
            "annotation",
            id="annotation_off_the_axis",
        ),
        pytest.param(
            [{**block("table"), "rows": [["only one cell"]]}],
            "columns",
            id="table_row_must_match_columns",
        ),
        pytest.param(
            [
                {
                    "type": "gallery",
                    "images": [{"src": "https://evil.io/a.png", "caption": "x"}],
                }
            ],
            "src",
            id="gallery_refuses_http_url",
        ),
        pytest.param(
            [
                {
                    "type": "gallery",
                    "images": [
                        {"src": "workspace://abc#text/html", "caption": "x"},
                    ],
                }
            ],
            "src",
            id="gallery_refuses_non_image",
        ),
    ],
)
async def test_invalid_blocks_are_refused_with_a_reason(
    tool: RenderUITool, session: ChatSession, bad_blocks: Any, expected: str
):
    result = await run(tool, session, bad_blocks)

    assert isinstance(result, ErrorResponse)
    assert expected in result.message


@pytest.mark.asyncio
async def test_null_optional_fields_are_accepted(
    tool: RenderUITool, session: ChatSession
):
    blocks = [
        {"type": "stats", "items": [{"label": "Runs", "value": "7", "change": None}]}
    ]

    result = await run(tool, session, blocks)

    assert isinstance(result, UIBlocksResponse)


@pytest.mark.asyncio
async def test_oversized_payload_is_refused(tool: RenderUITool, session: ChatSession):
    wide_row = ["x" * 200] * 6
    wide_table = {
        "type": "table",
        "columns": ["a", "b", "c", "d", "e", "f"],
        "rows": [wide_row] * 20,
    }

    result = await run(tool, session, [wide_table] * 6)

    assert isinstance(result, ErrorResponse)
    assert str(MAX_PAYLOAD_CHARS) in result.message


def test_parameters_are_flat_and_keep_title_fields():
    parameters = render_ui_parameters()
    text = json.dumps(parameters)

    for keyword in ("$ref", "$defs", "discriminator", "oneOf"):
        assert keyword not in text
    # A schema `title` keyword holds a string; a field called title holds an object.
    assert '"title": "' not in text
    assert parameters["required"] == ["blocks"]
    items = parameters["properties"]["blocks"]["items"]["anyOf"]
    chart = next(
        option
        for option in items
        if option["properties"]["type"].get("const") == "chart"
    )
    assert "title" in chart["properties"]
    # The SDK validates against this before the tool runs, and models send null
    # for optional fields they leave empty.
    assert chart["properties"]["title"]["type"] == ["string", "null"]
    stats = next(
        option
        for option in items
        if option["properties"]["type"].get("const") == "stats"
    )
    tone = stats["properties"]["items"]["items"]["properties"]["tone"]
    assert None in tone["enum"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, flag_on, hidden",
    [
        pytest.param({}, True, False, id="person_in_the_chat"),
        pytest.param({"origin": "automation"}, True, False, id="scheduled_follow_up"),
        pytest.param({}, False, True, id="flag_off"),
        pytest.param({"source_platform": "slack"}, True, True, id="chat_platform"),
        pytest.param({"text_only_result": True}, True, True, id="text_only_caller"),
        pytest.param(
            {"origin": "automation", "delegated_by_session_id": "parent"},
            True,
            True,
            id="thread_reporting_back",
        ),
        pytest.param(
            {
                "delegated_by_session_id": "parent",
                "handed_off_from_expert_id": "expert",
            },
            True,
            False,
            id="handed_off_thread",
        ),
    ],
)
async def test_render_ui_availability(
    overrides: dict[str, Any], flag_on: bool, hidden: bool
):
    session = ChatSession.new(user_id="test-user", dry_run=False, **overrides)

    with patch(
        "backend.copilot.tools.render_ui.is_feature_enabled",
        AsyncMock(return_value=flag_on),
    ):
        disabled = await render_ui_disabled_tools(session)

    assert (RENDER_UI_TOOL in disabled) is hidden


@pytest.mark.parametrize(
    "permissions",
    [
        pytest.param(None, id="no_filter"),
        pytest.param(CopilotPermissions(), id="empty_filter"),
        pytest.param(
            CopilotPermissions(tools=["bash_exec"], tools_exclude=True),
            id="deny_list",
        ),
        pytest.param(
            CopilotPermissions(tools=["render_ui", "web_fetch"], tools_exclude=False),
            id="allow_list",
        ),
    ],
)
def test_text_only_turns_refuse_render_ui(permissions: CopilotPermissions | None):
    all_tools = all_known_tool_names()
    before = (permissions or CopilotPermissions()).effective_allowed_tools(all_tools)

    after = without_render_ui(permissions).effective_allowed_tools(all_tools)

    assert RENDER_UI_TOOL not in after
    assert after == before - {RENDER_UI_TOOL}


@pytest.mark.asyncio
async def test_only_cards_count_as_shown(tool: RenderUITool, session: ChatSession):
    shown = await tool._execute(
        user_id=None, session=session, blocks=ALL_BLOCKS[:MAX_BLOCKS]
    )
    refused = await tool._execute(user_id=None, session=session, blocks=[])

    assert shows_cards(shown.model_dump_json(exclude_none=True))
    assert not shows_cards(refused.model_dump_json(exclude_none=True))
    assert not shows_cards("Tool execution error: boom")
