"""Database boundaries must fail before a Prisma-less worker is deployed."""

import json
from pathlib import Path

import pytest

from backend.util.db_boundary import check_database_boundary, find_violations

QUERY_SOURCE = """
from prisma.models import User

async def read_user(user_id):
    return await User.prisma().find_unique(where={"id": user_id})

class UserSummary:
    pass

def format_user(user):
    return str(user)
"""


@pytest.mark.parametrize(
    "source",
    [
        "from prisma.models import User\nUser.prisma().find_many()",
        "from prisma.models import User as U\naccessor = U.prisma\naccessor()",
        "from prisma import Prisma as Client\nClient()",
        "import prisma as orm\norm.Prisma()",
        "from backend.data.db import prisma as client\nclient.user.find_many()",
        "from backend.data.db import query_raw_with_schema as query\nquery('SELECT 1')",
        "from backend.data import db as database\ndatabase.prisma.user.find_many()",
        "import backend.data.db as database\ndatabase.transaction()",
        "from backend.data.user import read_user as load\nload('user')",
        "from backend.data import user as users\nusers.read_user('user')",
        "import backend.data.user\nbackend.data.user.read_user('user')",
        "from backend.data.user import *\nread_user('user')",
        "async def notify():\n    from backend.data.user import read_user\n    return await read_user('user')",
        "from backend.data import user\nfrom backend.data.db_accessors import user_db\nif connected:\n    selected = user\nelse:\n    selected = user_db()\nselected.read_user('user')",
        "from backend.data import user\nfrom backend.data.db_accessors import user_db\nselected = user if connected else user_db()\nselected.read_user('user')",
        "from backend.data.credit import get_user_credit_model\nget_user_credit_model('user')",
        "from backend.data import user\nfrom backend.data.user import UserSummary\nUserSummary(user.read_user('user')).label()",
    ],
)
def test_rejects_database_access_from_a_service(source: str):
    violations = find_violations(
        {"backend.data.user": QUERY_SOURCE, "backend.notifications.example": source}
    )
    assert any(key.startswith("notifications/example.py::") for key in violations)


def test_detects_transitive_helpers_and_reexports():
    sources = {
        "backend.data.user": QUERY_SOURCE,
        "backend.util.lookup": (
            "from backend.data.user import read_user as lookup\n"
            "async def user_name(user_id):\n    return await lookup(user_id)\n"
        ),
        "backend.util.public": "from backend.util.lookup import user_name as name",
        "backend.notifications.example": (
            "from backend.util.public import name\n"
            "async def notify():\n    return await name('user')\n"
        ),
    }
    assert any(
        key.startswith("notifications/example.py::") for key in find_violations(sources)
    )


@pytest.mark.parametrize(
    "source",
    [
        "from prisma.enums import SubscriptionTier\ntier = SubscriptionTier.FREE",
        "from prisma.models import User\ndef label(user: User):\n    return user.id",
        "from backend.data.user import UserSummary, format_user\nformat_user(UserSummary())",
        "from backend.data.db_accessors import user_db\nuser_db().get_user_by_id('user')",
        "from backend.util.clients import get_database_manager_async_client\nget_database_manager_async_client().get_user_by_id('user')",
        'message = "User.prisma().find_many()"\n# User.prisma().find_many()\n',
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from backend.data.user import read_user",
    ],
)
def test_allows_types_pure_helpers_and_routed_access(source: str):
    assert not find_violations(
        {"backend.data.user": QUERY_SOURCE, "backend.notifications.example": source}
    )


def test_relative_and_assignment_aliases_cannot_hide_queries():
    sources = {
        "backend.data.user": QUERY_SOURCE,
        "backend.notifications.lookup": (
            "from backend.data.user import read_user\nfetch = read_user\n"
        ),
        "backend.notifications.example": (
            "from .lookup import fetch as read\nread('user')\n"
        ),
    }
    assert any(
        key.startswith("notifications/example.py::") for key in find_violations(sources)
    )


def test_syntax_errors_are_not_silently_skipped():
    with pytest.raises(SyntaxError):
        find_violations({"backend.notifications.example": "async def broken("})


def test_backend_database_boundary():
    root = Path(__file__).resolve().parents[1]
    assert not (failures := check_database_boundary(root)), "\n".join(failures)


def test_sentry_workspace_embedding_bypass_is_rejected():
    sources = {
        "backend.api.features.search.embeddings": (
            "from backend.data.db import execute_raw_with_schema\n"
            "async def delete_content_embedding(content_type, content_id):\n"
            "    await execute_raw_with_schema('DELETE ...', content_type, content_id)\n"
        ),
        "backend.api.features.workspace.embeddings": (
            "from backend.api.features.search.embeddings import delete_content_embedding\n"
            "async def delete_workspace_file_embedding(file_id):\n"
            "    await delete_content_embedding('WORKSPACE_FILE', file_id)\n"
        ),
    }
    assert any(
        key.startswith("api/features/workspace/embeddings.py::")
        for key in find_violations(sources)
    )


def test_connection_router_cannot_hide_a_direct_query():
    violations = find_violations(
        {
            "backend.data.db_accessors": "from prisma.models import User\ndef user_db():\n    return User.prisma().find_many()"
        }
    )
    assert any(key.startswith("data/db_accessors.py::") for key in violations)


@pytest.mark.parametrize("name", ["is_connected", "cache_clear", "cache_delete"])
def test_query_helper_names_cannot_bypass_the_boundary(name: str):
    sources = {
        "backend.data.user": QUERY_SOURCE.replace("read_user", name),
        "backend.notifications.example": (
            f"from backend.data.user import {name}\n{name}('user')"
        ),
    }
    assert any(
        key.startswith("notifications/example.py::") for key in find_violations(sources)
    )


def test_connection_status_and_query_cache_invalidation_are_safe():
    sources = {
        "backend.data.user": QUERY_SOURCE,
        "backend.data.db": "from prisma import Prisma\nprisma = Prisma()\ndef is_connected():\n    return prisma.is_connected()",
        "backend.notifications.example": (
            "from backend.data import db, user\n"
            "db.is_connected()\nuser.read_user.cache_clear()"
        ),
    }
    assert not find_violations(sources)


def test_rpc_client_declarations_do_not_taint_their_callers():
    sources = {
        "backend.data.user": QUERY_SOURCE,
        "backend.example": (
            "from backend.util.service import AppServiceClient\n"
            "from backend.data.user import read_user\n"
            "class Client(AppServiceClient):\n    read_user = read_user\n"
        ),
        "backend.notifications.example": "from backend.example import Client\nClient().read_user('user')",
    }
    assert not any(
        key.startswith("notifications/example.py::") for key in find_violations(sources)
    )


def test_query_methods_do_not_make_model_types_unsafe_to_import():
    sources = {
        "backend.data.model": (
            "from prisma.models import User\n"
            "class Profile:\n"
            "    async def load(self):\n        return await User.prisma().find_many()\n"
        ),
        "backend.notifications.example": "from backend.data.model import Profile\ndef label(p: Profile):\n    return str(p)",
    }
    assert not find_violations(sources)


def test_forward_defined_helpers_and_package_reexports_are_checked():
    sources = {
        "backend.data.user": QUERY_SOURCE,
        "backend.util.lookup": (
            "from backend.data.user import read_user\n"
            "async def lookup():\n    return await later()\n"
            "async def later():\n    return await read_user('user')\n"
        ),
        "backend.util.public.__init__": "from backend.util.lookup import lookup",
        "backend.notifications.example": "from backend.util.public import lookup\nlookup()",
    }
    assert any(
        key.startswith("notifications/example.py::") for key in find_violations(sources)
    )


def test_module_and_export_with_the_same_name_do_not_loop():
    sources = {
        "backend.cli.main": "def main():\n    return 'ready'",
        "backend.cli.__init__": "from backend.cli.main import main",
    }
    assert not find_violations(sources)


def test_query_reexport_named_after_its_module_is_still_checked():
    sources = {
        "backend.data.user": "from prisma.models import User\nasync def user():\n    return await User.prisma().find_many()",
        "backend.data.__init__": "from backend.data.user import user",
        "backend.notifications.example": "from backend.data import user\nuser()",
    }
    assert any(
        key.startswith("notifications/example.py::") for key in find_violations(sources)
    )


def test_legacy_counts_cannot_hide_an_extra_call_or_a_later_regression(tmp_path: Path):
    source = "from prisma.models import User\nUser.prisma().find_many()\n"
    notification = tmp_path / "notifications" / "example.py"
    notification.parent.mkdir()
    notification.write_text(source, encoding="utf-8")
    baseline = tmp_path / "util" / "database_boundary_legacy.json"
    baseline.parent.mkdir()
    original = find_violations({"backend.notifications.example": source})
    baseline.write_text(
        json.dumps({key: len(lines) for key, lines in original.items()}),
        encoding="utf-8",
    )
    assert not check_database_boundary(tmp_path)

    notification.write_text(source + "User.prisma().find_many()\n", encoding="utf-8")
    assert any(
        "2 references; 1 legacy" in error for error in check_database_boundary(tmp_path)
    )

    notification.write_text("", encoding="utf-8")
    assert any("stale legacy" in error for error in check_database_boundary(tmp_path))
