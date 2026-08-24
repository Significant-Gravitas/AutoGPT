from types import SimpleNamespace

import pytest

from backend.copilot.partner_context import build_partner_system_prompt_suffix


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


def _session(source_platform: str | None, external_account_id: str | None):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            source_platform=source_platform,
            external_account_id=external_account_id,
        )
    )


def test_forwarding_digital_session_gets_mcp_instructions():
    result = build_partner_system_prompt_suffix(
        _session("forwarding-digital", "fd-account-77")
    )

    assert "query_forwarding_digital" in result
    assert "tenant-bound" in result
    assert "fd-account-77" not in result


def test_non_partner_session_gets_no_partner_instructions():
    assert build_partner_system_prompt_suffix(_session(None, None)) == ""


def test_partner_session_without_account_fails_closed():
    assert (
        build_partner_system_prompt_suffix(_session("forwarding-digital", None)) == ""
    )
