from types import SimpleNamespace

import pytest

from backend.copilot.partner_context import (
    build_partner_system_prompt_suffix,
    partner_disallowed_graph_block_ids,
    partner_session_has_capability,
)


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


def _session(
    source_platform: str | None,
    external_account_id: str | None,
    capabilities: list[str] | None = None,
):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            source_platform=source_platform,
            external_account_id=external_account_id,
            external_capabilities=capabilities or [],
        )
    )


def test_forwarding_digital_session_gets_mcp_instructions():
    result = build_partner_system_prompt_suffix(
        _session("forwarding-digital", "fd-account-77")
    )

    assert "query_forwarding_digital" in result
    assert "tenant-bound" in result
    assert "fd-account-77" not in result


def test_partner_lifecycle_prompt_matches_capabilities():
    manager = build_partner_system_prompt_suffix(
        _session(
            "forwarding-digital",
            "fd-account-77",
            ["agents.create", "agents.run", "agents.schedule"],
        )
    )
    operator = build_partner_system_prompt_suffix(
        _session("forwarding-digital", "fd-account-77", ["jobs.read"])
    )

    assert "enter_agent_building_mode" in manager
    assert "schedule_name" in manager
    assert "enter_agent_building_mode" not in operator


def test_non_partner_session_gets_no_partner_instructions():
    assert build_partner_system_prompt_suffix(_session(None, None)) == ""


def test_partner_session_without_account_fails_closed():
    assert (
        build_partner_system_prompt_suffix(_session("forwarding-digital", None)) == ""
    )


def test_partner_capability_check_fails_closed():
    session = _session(
        "forwarding-digital",
        "fd-account-77",
        ["agents.create"],
    )

    assert partner_session_has_capability(session, "agents.create")
    assert not partner_session_has_capability(session, "agents.schedule")
    assert partner_session_has_capability(_session(None, None), "agents.schedule")


def test_unknown_embedded_partner_defaults_to_denied():
    session = _session(
        "future-partner",
        "future-account",
        ["agents.create", "autogpt:block:allowed-block"],
    )

    assert partner_session_has_capability(session, "agents.create")
    assert not partner_session_has_capability(session, "agents.run")
    assert partner_disallowed_graph_block_ids(
        session, {"nodes": [{"block_id": "denied-block"}]}
    ) == ["denied-block"]


def test_partner_graph_blocks_include_nested_subgraphs():
    session = _session(
        "forwarding-digital",
        "fd-account-77",
        ["autogpt:block:allowed-block"],
    )
    graph = {
        "nodes": [{"block_id": "allowed-block"}],
        "sub_graphs": [
            {
                "nodes": [
                    {"block_id": "denied-block"},
                    {"block_id": "denied-block"},
                ]
            }
        ],
    }

    assert partner_disallowed_graph_block_ids(session, graph) == ["denied-block"]
