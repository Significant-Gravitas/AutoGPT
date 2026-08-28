from urllib.parse import parse_qs, urlparse

from backend.util.tenancy_urls import builder_path, copilot_path, library_agent_path


def _query(path: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(path).query)


def test_library_agent_path_encodes_identity_scope_and_run() -> None:
    path = library_agent_path(
        "library/agent",
        "org one",
        "team/two",
        active_tab="runs",
        active_item="exec/three",
    )

    assert urlparse(path).path == "/library/agents/library%2Fagent"
    assert _query(path) == {
        "organizationId": ["org one"],
        "teamId": ["team/two"],
        "activeTab": ["runs"],
        "activeItem": ["exec/three"],
    }


def test_personal_and_org_home_scope_use_explicit_sentinels() -> None:
    assert _query(copilot_path("session", None, None)) == {
        "organizationId": ["__personal__"],
        "teamId": ["__org_home__"],
        "sessionId": ["session"],
    }


def test_builder_path_preserves_exact_execution_scope() -> None:
    path = builder_path(
        "graph",
        7,
        "org",
        "team",
        execution_id="execution",
    )

    assert urlparse(path).path == "/build"
    assert _query(path) == {
        "organizationId": ["org"],
        "teamId": ["team"],
        "flowID": ["graph"],
        "flowVersion": ["7"],
        "flowExecutionID": ["execution"],
    }
