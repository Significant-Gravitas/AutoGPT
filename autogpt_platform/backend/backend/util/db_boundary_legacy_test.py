import json
from pathlib import Path

import pytest

from backend.util.db_boundary import (
    check_database_boundary,
    find_database_references,
    find_violations,
)

SOURCE = (
    "from prisma.models import User, AgentGraph\n"
    "async def query(user_id):\n"
    "    return await User.prisma().find_unique(where={'id': user_id})\n"
)


def _baseline(root: Path, source: str = SOURCE) -> Path:
    path = root / "notifications" / "example.py"
    path.parent.mkdir()
    path.write_text(source, encoding="utf-8")
    ledger = root / "util" / "database_boundary_legacy.json"
    ledger.parent.mkdir()
    references = find_database_references({"backend.notifications.example": source})
    ledger.write_text(
        json.dumps(
            {
                key: [identity for _, identity in refs]
                for key, refs in references.items()
            }
        ),
        encoding="utf-8",
    )
    assert not check_database_boundary(root)
    return path


@pytest.mark.parametrize(
    "replacement",
    [
        SOURCE.replace("find_unique", "delete"),
        SOURCE.replace("User.prisma()", "AgentGraph.prisma()"),
        SOURCE.replace("{'id': user_id}", "{'id': 'another-user'}"),
    ],
)
def test_equal_count_query_replacement_is_rejected(tmp_path: Path, replacement: str):
    path = _baseline(tmp_path)
    original = find_violations({"backend.notifications.example": SOURCE})
    assert original == find_violations({"backend.notifications.example": replacement})
    path.write_text(replacement, encoding="utf-8")
    assert check_database_boundary(tmp_path)


def test_formatting_and_line_movement_do_not_replace_a_reference(tmp_path: Path):
    path = _baseline(tmp_path)
    path.write_text(
        "# A comment before the imports.\n\n"
        + SOURCE.replace(
            "find_unique(where={'id': user_id})",
            'find_unique(\n        where={"id": user_id},\n    )',
        ),
        encoding="utf-8",
    )
    assert not check_database_boundary(tmp_path)


def test_import_alias_rename_does_not_replace_a_reference(tmp_path: Path):
    path = _baseline(tmp_path)
    path.write_text(
        SOURCE.replace("import User,", "import User as Account,").replace(
            "User.prisma()", "Account.prisma()"
        ),
        encoding="utf-8",
    )
    assert not check_database_boundary(tmp_path)


def test_references_take_the_identity_of_their_innermost_call():
    key = "notifications/example.py::query -> .prisma"
    nested = find_database_references(
        {
            "backend.notifications.example": (
                "from prisma.models import AgentGraph, User\n"
                "async def query():\n"
                "    return pair(User.prisma().find_many(), AgentGraph.prisma().find_many())\n"
                "def pair(*items):\n    return items\n"
            )
        }
    )
    direct = find_database_references(
        {
            "backend.notifications.example": (
                "from prisma.models import AgentGraph, User\n"
                "async def query():\n"
                "    await User.prisma().find_many()\n"
                "    await AgentGraph.prisma().find_many()\n"
            )
        }
    )
    identities = [identity for _, identity in nested[key]]
    assert len(set(identities)) == 2
    assert identities == [identity for _, identity in direct[key]]


def test_equal_count_replacement_checks_identity_multiplicity(tmp_path: Path):
    query = "User.prisma().find_unique(where={'id': user_id})"
    source = SOURCE.replace(
        f"return await {query}",
        f"await {query}\n    await {query}\n    return await "
        "AgentGraph.prisma().find_unique(where={'id': user_id})",
    )
    path = _baseline(tmp_path, source)
    replacement = source.replace(
        f"await {query}\n",
        "await " + query.replace("User.prisma()", "AgentGraph.prisma()") + "\n",
        1,
    )
    assert find_violations(
        {"backend.notifications.example": source}
    ) == find_violations({"backend.notifications.example": replacement})
    path.write_text(replacement, encoding="utf-8")
    assert check_database_boundary(tmp_path)
