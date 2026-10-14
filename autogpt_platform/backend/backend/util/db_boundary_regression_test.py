import ast
from pathlib import Path

import pytest

from backend.util.db_boundary import find_violations
from backend.util.db_boundary_policy import CLI_DISPATCHER, CONNECTION_OWNERS


def test_pure_nested_helpers_do_not_inherit_the_parent_query():
    sources = {
        "backend.notifications.example": (
            "from prisma.models import User\n"
            "async def route():\n"
            "    def helper():\n        return 'label'\n"
            "    await User.prisma().find_many()\n"
            "    helper()\n    helper()\n"
        )
    }
    violations = find_violations(sources)
    assert violations == {"notifications/example.py::route -> .prisma": [5]}


def test_nested_query_helpers_still_taint_their_callers():
    sources = {
        "backend.notifications.example": (
            "from prisma.models import User\n"
            "async def route():\n"
            "    async def helper():\n        return await User.prisma().find_many()\n"
            "    return await helper()\n"
        )
    }
    assert (
        "notifications/example.py::route -> backend.notifications.example.route.helper"
        in find_violations(sources)
    )


CONNECTION_OWNER_SOURCE = (
    "from backend.data.db import connect, disconnect, query_raw_with_schema\n"
    "async def read_records():\n    return await query_raw_with_schema('SELECT 1')\n"
    "async def run():\n"
    "    await connect()\n"
    "    try:\n        return await read_records()\n"
    "    finally:\n        await disconnect()\n"
    "def onboarding_role_backfill_command():\n    return run()\n"
)


def test_reviewed_connection_owner_and_its_cli_dispatch_are_allowed():
    assert not find_violations(
        {
            "backend.cli.onboarding_role_backfill": CONNECTION_OWNER_SOURCE,
            "backend.cli.main": (
                "from backend.cli.onboarding_role_backfill import onboarding_role_backfill_command\n"
                "main.add_command(onboarding_role_backfill_command)\n"
            ),
        }
    )


def test_unreviewed_cli_cannot_claim_connection_ownership():
    assert find_violations({"backend.cli.unreviewed": CONNECTION_OWNER_SOURCE})


@pytest.mark.parametrize(
    "caller, source",
    [
        (
            "backend.notifications.example",
            "from backend.cli.onboarding_role_backfill import read_records\nread_records()",
        ),
        (
            "backend.notifications.example",
            "from backend.cli.onboarding_role_backfill import onboarding_role_backfill_command\nonboarding_role_backfill_command()",
        ),
        (
            "backend.cli.main",
            "from backend.cli.onboarding_role_backfill import read_records\nread_records()",
        ),
        (
            "backend.cli.main",
            "from backend.data.db import connect\nconnect()",
        ),
    ],
)
def test_connection_owner_helpers_and_dispatcher_do_not_exempt_callers(
    caller: str, source: str
):
    violations = find_violations(
        {
            "backend.cli.onboarding_role_backfill": CONNECTION_OWNER_SOURCE,
            caller: source,
        }
    )
    path = caller.removeprefix("backend.").replace(".", "/") + ".py::"
    assert any(key.startswith(path) for key in violations)


@pytest.mark.parametrize("module", sorted(CONNECTION_OWNERS))
def test_connection_owner_approvals_name_existing_commands(module: str):
    root = Path(__file__).resolve().parents[1]
    path = root.joinpath(*module.removeprefix("backend.").split(".")).with_suffix(".py")
    assert path.is_file(), f"Remove stale connection owner: {module}"
    names = {
        node.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    assert CONNECTION_OWNERS[module] <= names
    dispatcher = root.joinpath(
        *CLI_DISPATCHER.removeprefix("backend.").split(".")
    ).with_suffix(".py")
    assert dispatcher.is_file(), "Remove stale CLI dispatcher approval"
