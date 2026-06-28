"""Tests for resolve_agent_json_input — the OPEN-3188 file-reference fallback.

The full agent graph is the largest argument the assistant emits inline, so it
is frequently truncated/dropped on the SDK path. These tests pin the two ways a
graph can now reach a tool: inline ``agent_json`` and a ``agent_json_ref``
pointing at a workspace file.
"""

import pytest

from backend.copilot.tools.agent_json_input import (
    AGENT_JSON_SCHEMA,
    resolve_agent_json_input,
)

from ._test_data import make_session

_USER_ID = "test-user-agent-json-input"
_MODULE = "backend.copilot.tools.agent_json_input"

_GRAPH = {"nodes": [{"id": "n1"}], "links": []}


def test_schema_is_structured_not_bare_object():
    """The schema must describe nodes/links so decoders don't collapse it to {}."""
    props = AGENT_JSON_SCHEMA.get("properties", {})
    assert "nodes" in props
    assert "links" in props
    assert AGENT_JSON_SCHEMA.get("additionalProperties") is True


@pytest.mark.asyncio
async def test_inline_object_wins():
    session = make_session(_USER_ID)
    graph, err = await resolve_agent_json_input(_GRAPH, None, _USER_ID, session)
    assert err is None
    assert graph == _GRAPH


@pytest.mark.asyncio
async def test_inline_stringified_json_is_parsed():
    """Some models emit the object as a JSON string — accept that too."""
    session = make_session(_USER_ID)
    graph, err = await resolve_agent_json_input(
        '{"nodes": [{"id": "n1"}], "links": []}', None, _USER_ID, session
    )
    assert err is None
    assert graph == _GRAPH


@pytest.mark.asyncio
async def test_ref_already_expanded_to_text():
    """SDK path: the file-ref wrapper already turned the token into file text."""
    session = make_session(_USER_ID)
    graph, err = await resolve_agent_json_input(
        None, '{"nodes": [{"id": "n1"}], "links": []}', _USER_ID, session
    )
    assert err is None
    assert graph == _GRAPH


@pytest.mark.asyncio
async def test_ref_workspace_uri_is_read(mocker):
    """Baseline path: a plain workspace URI is read server-side and parsed."""
    session = make_session(_USER_ID)
    read = mocker.patch(
        f"{_MODULE}.read_file_bytes",
        return_value=b'{"nodes": [{"id": "n1"}], "links": []}',
    )
    graph, err = await resolve_agent_json_input(
        None, "workspace:///agent.json", _USER_ID, session
    )
    assert err is None
    assert graph == _GRAPH
    read.assert_awaited_once()
    assert read.call_args.args[0] == "workspace:///agent.json"


@pytest.mark.asyncio
async def test_ref_agptfile_token_is_read(mocker):
    """An unexpanded @@agptfile: token resolves to its URI before reading."""
    session = make_session(_USER_ID)
    read = mocker.patch(
        f"{_MODULE}.read_file_bytes",
        return_value=b'{"nodes": [{"id": "n1"}], "links": []}',
    )
    graph, err = await resolve_agent_json_input(
        None, "@@agptfile:workspace:///agent.json", _USER_ID, session
    )
    assert err is None
    assert graph == _GRAPH
    assert read.call_args.args[0] == "workspace:///agent.json"


@pytest.mark.asyncio
async def test_bare_filename_becomes_workspace_path(mocker):
    session = make_session(_USER_ID)
    read = mocker.patch(
        f"{_MODULE}.read_file_bytes",
        return_value=b'{"nodes": [{"id": "n1"}], "links": []}',
    )
    await resolve_agent_json_input(None, "agent.json", _USER_ID, session)
    assert read.call_args.args[0] == "workspace:///agent.json"


@pytest.mark.asyncio
async def test_nothing_provided_returns_none_none():
    session = make_session(_USER_ID)
    graph, err = await resolve_agent_json_input(None, None, _USER_ID, session)
    assert graph is None
    assert err is None


@pytest.mark.asyncio
async def test_ref_file_read_error_surfaces_message(mocker):
    session = make_session(_USER_ID)
    mocker.patch(
        f"{_MODULE}.read_file_bytes",
        side_effect=ValueError("File not found: workspace:///missing.json"),
    )
    graph, err = await resolve_agent_json_input(
        None, "workspace:///missing.json", _USER_ID, session
    )
    assert graph is None
    assert err is not None
    assert "File not found" in err


@pytest.mark.asyncio
async def test_ref_file_non_object_errors(mocker):
    session = make_session(_USER_ID)
    mocker.patch(f"{_MODULE}.read_file_bytes", return_value=b"[1, 2, 3]")
    graph, err = await resolve_agent_json_input(
        None, "workspace:///agent.json", _USER_ID, session
    )
    assert graph is None
    assert err is not None
    assert "JSON object" in err
