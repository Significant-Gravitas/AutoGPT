"""The notification service owns no database connection.

`AppService.lifespan` connects Redis and leaves Prisma to subclasses; only
`rest_api`, `ws_api` and `db_manager` call `db.connect()`. Everything the
NotificationManager process runs therefore has to reach the database through
the DatabaseManager RPC.

A grep is a blunt instrument, but it fails at the moment the boundary is
crossed rather than the first time the service is deployed.
"""

import ast
import pathlib

import pytest

# Modules in this package that do NOT run inside the notification service, and
# so are allowed their own Prisma access. Each is called from a process that
# owns a connection; adding to this list means asserting the same.
_RUNS_ELSEWHERE = {
    # Called from data/execution.py, which is exposed on the DatabaseManager and
    # therefore runs inside it.
    "scoring.py",
    # Called from data/human_review.py, likewise inside the DatabaseManager.
    "review_alerts.py",
}

_FORBIDDEN_CALLS = {"prisma"}
_FORBIDDEN_IMPORTS = {
    "backend.data.db": {"prisma", "query_raw_with_schema", "connect"},
}

_PACKAGE = pathlib.Path(__file__).parent


def _modules() -> list[pathlib.Path]:
    return sorted(
        p
        for p in _PACKAGE.glob("*.py")
        if not p.name.endswith("_test.py") and p.name not in _RUNS_ELSEWHERE
    )


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_no_direct_prisma_access_in_the_notification_service(module: pathlib.Path):
    tree = ast.parse(module.read_text(), filename=str(module))

    for node in ast.walk(tree):
        # `Something.prisma()` — the Prisma model accessor.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _FORBIDDEN_CALLS
        ):
            pytest.fail(
                f"{module.name}:{node.lineno} calls .prisma() directly. The "
                "notification service has no Prisma connection — go through "
                "get_database_manager_async_client() instead."
            )

        if isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_IMPORTS:
            banned = _FORBIDDEN_IMPORTS[node.module] & {a.name for a in node.names}
            if banned:
                pytest.fail(
                    f"{module.name}:{node.lineno} imports {sorted(banned)} from "
                    f"{node.module}. The notification service has no Prisma "
                    "connection — go through the DatabaseManager RPC instead."
                )


def test_the_allowlist_only_names_modules_that_exist():
    """A stale allowlist entry would silently exempt nothing, or worse, mask a
    module that later moved into the service."""
    for name in _RUNS_ELSEWHERE:
        assert (_PACKAGE / name).exists(), f"{name} is allowlisted but does not exist"


def test_the_service_does_not_connect_a_database_itself():
    """If this ever needs changing, the RPC boundary has been abandoned and
    every module above is free to query Prisma again — which is the state that
    shipped the outage."""
    source = (_PACKAGE / "notifications.py").read_text()
    assert "db.connect()" not in source, (
        "NotificationManager must not open its own Prisma connection; its "
        "database access belongs behind the DatabaseManager RPC."
    )


def test_no_rpc_endpoint_is_aliased():
    """A renamed RPC endpoint is a 404 waiting to happen.

    `AppService` registers routes by *attribute* name
    (``route_path = f"/{attr_name}"``) while the client builds its URL from the
    underlying function's ``__name__`` (``rpc_name = original_func.__name__``).
    Alias one and the server exposes `/get_pending_alert_conditions` while the
    client POSTs `/get_pending_conditions`.

    This is silent until something actually calls the endpoint: the two aliases
    the email redesign introduced were dead for as long as its callers used
    Prisma directly, and only 404'd once they went through the RPC.
    """
    from backend.data.db_manager import DatabaseManager

    aliased = []
    for name, attr in vars(DatabaseManager).items():
        if name.startswith("_"):
            continue
        func = getattr(attr, "__wrapped__", attr)
        real = getattr(func, "__name__", None)
        if real and real not in (name, "_stub"):
            aliased.append(f"/{name} is served, but clients POST /{real}")

    assert not aliased, "aliased RPC endpoints 404 at runtime:\n  " + "\n  ".join(
        aliased
    )
