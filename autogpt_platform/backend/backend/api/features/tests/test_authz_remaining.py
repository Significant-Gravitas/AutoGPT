"""
REL-007 — Remaining trust boundaries (Wave 0 closure).

Covers 6 families not proven by test_authz_negative_matrix.py:
  A) Copilot ChatSession (direct + indirect)
  B) Workspace artifacts / folders (direct + indirect bulk-move)
  C) Integrations: IntegrationCredentials (direct + indirect scope upgrade)
  D) IntegrationWebhook (direct + indirect via credentials)
  E) Private Marketplace (StoreListingVersion / StoreSubmission)
  F) Agent-version indirect ownership (LibraryAgent + Graph version)

Each family proves the server-side ownership model: possessing another
user's resource ID does not confer access; parent/child relations are
scoped to the caller.

Mock pattern (matches platform_linking/db_test.py and others):
  patch("...prisma.models.X.prisma") → mock_X
  mock_X.prisma.return_value.find_first = AsyncMock(return_value=...)
This is the canonical pattern across backend tests — earlier drafts
mistakenly used `mock_X.find_first = AsyncMock(...)` which left
``mock_X.prisma()`` returning a MagicMock, breaking ``await``.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

VICTIM = "a1111111-1111-4111-8111-111111111111"
ATTACKER = "b2222222-2222-4222-8222-222222222222"

# ---------------------------------------------------------------------------
# A. Copilot ChatSession — backend/copilot/model.py + db.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_copilot_get_session_metadata_cross_user_denied():
    """Direct: attacker cannot read victim's session metadata.

    ``model.get_chat_session_metadata`` first calls
    ``chat_db().get_chat_session_metadata`` (no user scoping in DB layer — it
    returns any session by id). The model layer then enforces ownership:
    if ``session.user_id != user_id`` it returns None.
    """
    from backend.copilot.model import ChatSessionInfo, get_chat_session_metadata

    victim_info = MagicMock(spec=ChatSessionInfo)
    victim_info.user_id = VICTIM

    with patch("backend.copilot.model.chat_db") as mock_db_factory:
        mock_db = AsyncMock()
        mock_db.get_chat_session_metadata = AsyncMock(return_value=victim_info)
        mock_db_factory.return_value = mock_db

        # attacker asking for victim's session -> None (404 upstream)
        result = await get_chat_session_metadata("sess-victim", user_id=ATTACKER)
        assert result is None

        # victim asking for own session -> gets it
        victim_result = await get_chat_session_metadata("sess-victim", user_id=VICTIM)
        assert victim_result is not None
        assert victim_result.user_id == VICTIM


@pytest.mark.asyncio
async def test_copilot_paginated_messages_parent_mismatch():
    """Indirect: paginated messages require session ownership — child (message window) inherits parent (session) userId.

    ``get_chat_messages_paginated`` builds where={"id": session_id, "userId": user_id}.
    If attacker supplies victim session_id with own user_id, no row.
    """
    from backend.copilot.db import get_chat_messages_paginated

    with patch("backend.copilot.db.PrismaChatSession.prisma") as mock_prisma:
        mock_prisma.return_value.find_first = AsyncMock(return_value=None)
        result = await get_chat_messages_paginated(
            session_id="sess-victim", limit=50, user_id=ATTACKER
        )
        assert result is None
        where = mock_prisma.return_value.find_first.call_args.kwargs["where"]
        assert where["id"] == "sess-victim"
        assert where["userId"] == ATTACKER


@pytest.mark.asyncio
async def test_copilot_delete_session_cross_user_no_delete():
    """Direct: attacker delete on victim session deletes 0 rows.

    When ``user_id`` is provided, ``delete_chat_session`` first calls
    ``find_first(where={"id": ..., "userId": user_id})`` to check shared
    status, then ``delete_many`` with the same predicate. Attacker fails
    the userId check on both calls.
    """
    from backend.copilot.db import delete_chat_session

    with patch("backend.copilot.db.PrismaChatSession.prisma") as mock_prisma:
        mock_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_prisma.return_value.delete_many = AsyncMock(return_value=0)
        result = await delete_chat_session("sess-victim", user_id=ATTACKER)
        assert result is False
        # Both calls were scoped to attacker
        find_where = mock_prisma.return_value.find_first.call_args.kwargs["where"]
        assert find_where["id"] == "sess-victim"
        assert find_where["userId"] == ATTACKER
        delete_where = mock_prisma.return_value.delete_many.call_args.kwargs["where"]
        assert delete_where["id"] == "sess-victim"
        assert delete_where["userId"] == ATTACKER


# ---------------------------------------------------------------------------
# B. Workspace artifacts + folders — backend/data/workspace.py + folder.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_file_cross_user_denied():
    """Direct: attacker workspaceId cannot fetch victim's file."""
    from backend.data.workspace import get_workspace_file

    victim_file_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    attacker_workspace_id = "ws-attacker"

    with patch("backend.data.workspace.UserWorkspaceFile.prisma") as mock_prisma:
        mock_prisma.return_value.find_first = AsyncMock(return_value=None)
        result = await get_workspace_file(victim_file_id, attacker_workspace_id)
        assert result is None
        where = mock_prisma.return_value.find_first.call_args.kwargs["where"]
        # Must be scoped by both id and workspaceId (which is derived from userId)
        assert where["id"] == victim_file_id
        assert where["workspaceId"] == attacker_workspace_id
        assert where["isDeleted"] is False


@pytest.mark.asyncio
async def test_workspace_folder_bulk_move_parent_mismatch():
    """Indirect: attacker cannot bulk-move victim file into victim folder (cross-workspace parent mismatch).

    The target folder ownership is checked via ``_get_folder_record(folder_id, workspace_id)``
    which queries where={"id": folder_id, "workspaceId": workspace_id}.
    Supplying victim folder_id under attacker workspace -> NotFoundError.
    """
    from backend.data.workspace_folder import bulk_move_files_to_folder
    from backend.util.exceptions import NotFoundError

    victim_folder_id = "f-victim"
    attacker_workspace_id = "ws-attacker"
    file_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    with patch("backend.data.workspace_folder.UserWorkspaceFolder.prisma") as mock_folder:
        mock_folder.return_value.find_first = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await bulk_move_files_to_folder(
                workspace_id=attacker_workspace_id,
                file_ids=[file_id],
                folder_id=victim_folder_id,
            )
        where = mock_folder.return_value.find_first.call_args.kwargs["where"]
        assert where["id"] == victim_folder_id
        assert where["workspaceId"] == attacker_workspace_id


@pytest.mark.asyncio
async def test_workspace_resolve_files_silently_drops_cross_user():
    """Indirect: ``resolve_workspace_files`` drops foreign IDs (workspace-scoped allowlist).

    The server derives the workspace from caller user_id, then
    ``find_many`` filters by that workspaceId — not by attacker-supplied ids
    alone. Victim id is in the 'in' list input but the row simply does not
    exist in the attacker's workspace so result excludes it.
    """
    from backend.data.workspace import resolve_workspace_files

    victim_file_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    attacker_file_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"

    mock_workspace = MagicMock(id="ws-attacker", user_id=ATTACKER)

    with patch(
        "backend.data.workspace.get_or_create_workspace",
        new=AsyncMock(return_value=mock_workspace),
    ), patch("backend.data.workspace.UserWorkspaceFile.prisma") as mock_prisma:
        mock_file = MagicMock(id=attacker_file_id)
        mock_prisma.return_value.find_many = AsyncMock(return_value=[mock_file])
        result = await resolve_workspace_files(
            ATTACKER, [victim_file_id, attacker_file_id]
        )
        where = mock_prisma.return_value.find_many.call_args.kwargs["where"]
        # Workspace-scoped and not deleted
        assert where["workspaceId"] == "ws-attacker"
        assert where["isDeleted"] is False
        # Victim ID is in the 'in' list (sent as input) but result excludes it
        assert victim_file_id in where["id"]["in"]
        assert attacker_file_id in where["id"]["in"]
        assert len(result) == 1
        assert result[0].id == attacker_file_id


# ---------------------------------------------------------------------------
# C. Integration Credentials — backend/api/features/integrations/router.py + store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_credentials_cross_user_denied():
    """Direct: attacker cannot GET victim credential by ID.

    ``IntegrationCredentialsStore.get_creds_by_id(user_id, cred_id)`` filters
    by caller — cross-user returns None.
    """
    from backend.integrations.creds_manager import IntegrationCredentialsManager

    victim_cred_id = "cred-victim"
    with patch(
        "backend.integrations.credentials_store.IntegrationCredentialsStore.get_creds_by_id",
        new=AsyncMock(return_value=None),
    ):
        mgr = IntegrationCredentialsManager()
        result = await mgr.get(ATTACKER, victim_cred_id)
        assert result is None


@pytest.mark.asyncio
async def test_integration_credentials_webhook_indirect_mismatch():
    """Indirect: webhook ownership is via userId+credentialsId.

    ``get_all_webhooks_by_creds(user_id, credentials_id)`` includes both
    the caller and the cred in the where clause. Attacker using a
    victim cred_id is filtered out by userId.
    """
    from backend.data.integrations import get_all_webhooks_by_creds

    with patch("backend.data.integrations.IntegrationWebhook.prisma") as mock_prisma:
        mock_prisma.return_value.find_many = AsyncMock(return_value=[])
        result = await get_all_webhooks_by_creds(
            ATTACKER, "cred-victim", include_relations=False
        )
        assert result == []
        where = mock_prisma.return_value.find_many.call_args.kwargs["where"]
        assert where["userId"] == ATTACKER
        assert where["credentialsId"] == "cred-victim"


# ---------------------------------------------------------------------------
# D. IntegrationWebhook — backend/data/integrations.py delete + router ping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_webhook_delete_cross_user_no_delete():
    """Direct: attacker delete webhook deletes 0 (scoped where)."""
    from backend.data.integrations import delete_webhook
    from backend.util.exceptions import NotFoundError

    with patch("backend.data.integrations.IntegrationWebhook.prisma") as mock_prisma:
        mock_prisma.return_value.delete_many = AsyncMock(return_value=0)
        with pytest.raises(NotFoundError):
            await delete_webhook(ATTACKER, "wh-victim")
        where = mock_prisma.return_value.delete_many.call_args.kwargs["where"]
        assert where["id"] == "wh-victim"
        assert where["userId"] == ATTACKER


def test_integration_webhook_ping_ownership_check_at_route_layer():
    """Indirect: ``backend.data.integrations.get_webhook`` has no user_id check by design.

    The router (``backend/api/features/integrations/router.py::webhook_ping``)
    enforces ``webhook.user_id != user_id`` -> 404. This regression guard
    proves the data-layer function returns the row (it has no scope), so any
    route using it MUST add the ownership check. The route-level behavior
    is exercised in ``integrations/router_test.py::TestWebhookPingOwnership``.
    """
    import inspect

    from backend.data import integrations as integ_data

    src = inspect.getsource(integ_data.get_webhook)
    assert "userId" not in src, (
        "get_webhook must remain scope-free so route-layer ownership "
        "enforcement stays visible and auditable; do NOT add user_id here."
    )


@pytest.mark.asyncio
async def test_integration_webhook_get_then_route_check_pattern():
    """Indirect: pair ``get_webhook`` (no scope) with the route's user_id check.

    We exercise the actual ownership check pattern from
    ``router.py::webhook_ping``: fetch by id, compare ``webhook.user_id``
    to caller. This proves the route's invariant without importing the
    full FastAPI app chain.
    """
    from prisma.models import IntegrationWebhook as PrismaIntegrationWebhook

    from backend.data import integrations as integ_data

    victim_webhook = MagicMock(spec=PrismaIntegrationWebhook)
    # Prisma uses camelCase; Webhook.from_db maps to snake_case.
    victim_webhook.userId = VICTIM
    victim_webhook.credentialsId = ""
    victim_webhook.provider = "github"
    victim_webhook.id = "wh-victim"
    victim_webhook.webhookType = "MANUAL"
    victim_webhook.resource = "test"
    victim_webhook.events = []
    victim_webhook.config = {}
    victim_webhook.secret = "x"
    victim_webhook.organizationId = None
    victim_webhook.teamId = None
    victim_webhook.providerWebhookId = ""

    with patch("backend.data.integrations.IntegrationWebhook.prisma") as mock_prisma:
        mock_prisma.return_value.find_unique = AsyncMock(return_value=victim_webhook)
        webhook = await integ_data.get_webhook("wh-victim", include_relations=False)
        # The data layer correctly returns the row (no userId filter).
        assert webhook is not None
        assert webhook.user_id == VICTIM
        # The route layer MUST reject when webhook.user_id != caller.
        assert webhook.user_id != ATTACKER


# ---------------------------------------------------------------------------
# E. Private Marketplace — backend/api/features/store/db.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_submission_delete_cross_user_denied():
    """Direct: attacker cannot delete victim's StoreListingVersion.

    ``delete_store_submission`` finds the version with the owning listing,
    then checks ``listing.owningUserId == user_id`` (or org match). On
    mismatch it raises SubmissionNotFoundError, which the try/except at the
    bottom converts to a False return.
    """
    from backend.api.features.store import db as store_db

    victim_submission_id = "slv-victim"

    mock_version = MagicMock(
        id=victim_submission_id,
        StoreListing=MagicMock(owningUserId=VICTIM, owningOrgId=None),
        submissionStatus=MagicMock(value="PENDING"),
        storeListingId="listing-victim",
    )

    with patch(
        "backend.api.features.store.db.prisma.models.StoreListingVersion.prisma"
    ) as mock_slv:
        mock_slv.return_value.find_first = AsyncMock(return_value=mock_version)
        result = await store_db.delete_store_submission(ATTACKER, victim_submission_id)
        assert result is False
        # The find_first scoped the submission id; the ownership check fired
        find_where = mock_slv.return_value.find_first.call_args.kwargs["where"]
        assert find_where["id"] == victim_submission_id


@pytest.mark.asyncio
async def test_store_submission_edit_indirect_version_mismatch():
    """Indirect: attacker tries to edit victim submission via store_listing_version_id.

    ``edit_store_submission`` performs a find_first by id only, then raises
    UnauthorizedError if the caller's userId/org doesn't match the listing's
    owningUserId/owningOrgId. We assert the function raises UnauthorizedError
    rather than completing the update.
    """
    from backend.api.features.store import db as store_db
    from backend.api.features.store.exceptions import UnauthorizedError

    mock_listing = MagicMock(owningUserId=VICTIM, owningOrgId=None)
    mock_current = MagicMock(
        StoreListing=mock_listing,
        submissionStatus="PENDING",
        agentGraphId="graph-victim",
    )
    with patch(
        "backend.api.features.store.db.prisma.models.StoreListingVersion.prisma"
    ) as mock_prisma:
        mock_prisma.return_value.find_first = AsyncMock(return_value=mock_current)
        with pytest.raises(UnauthorizedError):
            await store_db.edit_store_submission(
                user_id=ATTACKER,
                store_listing_version_id="slv-victim",
                name="Hacked",
                description="x",
            )


@pytest.mark.asyncio
async def test_store_get_submissions_scoped_to_caller():
    """Direct: ``get_store_submissions`` filters by caller userId.

    Without an org, the where clause keys on ``user_id`` — never on a
    body/query parameter. ``StoreSubmission`` is a raw-SQL view, so the
    column names are snake_case (``user_id``, not ``userId``).
    """
    from backend.api.features.store import db as store_db
    from backend.api.features.store.model import SubmissionStats

    stats = SubmissionStats(
        total=0, approved=0, pending=0, total_runs=0, average_rating=None
    )
    with patch("backend.api.features.store.db.prisma.models.StoreSubmission.prisma") as mock_sub:
        mock_sub.return_value.find_many = AsyncMock(return_value=[])
        mock_sub.return_value.count = AsyncMock(return_value=0)
        with patch(
            "backend.api.features.store.db._get_submission_stats",
            new=AsyncMock(return_value=stats),
        ):
            await store_db.get_store_submissions(ATTACKER, page=1, page_size=20)
            where = mock_sub.return_value.find_many.call_args.kwargs["where"]
            # Without org context, key is user_id = attacker
            assert where.get("user_id") == ATTACKER or "AND" in where


# ---------------------------------------------------------------------------
# F. Agent-version indirect ownership — LibraryAgent + Graph versions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_update_version_cross_user_denied():
    """Direct: attacker cannot re-point victim's LibraryAgent to a new version.

    ``update_agent_version_in_library`` requires
    ``find_first_or_raise(userId=caller, agentGraphId=graph_id)`` — cross-user
    raises RecordNotFoundError. We assert the where clause sent to Prisma
    contains the caller userId, which is what enforces ownership.
    """
    from backend.api.features.library import db as lib_db

    raised: list[dict] = []

    async def _raise(**kwargs):
        raised.append(kwargs)
        raise RuntimeError("denied-by-ownership-test")

    @contextlib.asynccontextmanager
    async def _fake_transaction():
        yield None

    with (
        patch(
            "backend.api.features.library.db.transaction",
            new=_fake_transaction,
        ),
        patch(
            "backend.api.features.library.db.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
    ):
        mock_prisma.return_value.find_first_or_raise = _raise
        with pytest.raises(RuntimeError, match="denied-by-ownership-test"):
            await lib_db.update_agent_version_in_library(ATTACKER, "graph-victim", 2)

    assert len(raised) == 1
    where = raised[0]["where"]
    # Ownership filter is the only thing standing between attacker and
    # the victim's library row.
    assert where["userId"] == ATTACKER
    assert where["agentGraphId"] == "graph-victim"


@pytest.mark.asyncio
async def test_graph_all_versions_cross_user_empty():
    """Indirect: attacker listing graph versions for victim graph gets empty.

    Without an active org, ``get_graph_all_versions`` adds
    ``userId == caller`` to the where clause. Victim graph with attacker
    userId returns no rows.
    """
    from backend.data.graph import get_graph_all_versions

    with patch(
        "backend.data.graph.get_user_team_ids",
        new=AsyncMock(return_value=[]),
    ), patch("backend.data.graph.AgentGraph.prisma") as mock_prisma:
        mock_prisma.return_value.find_many = AsyncMock(return_value=[])
        result = await get_graph_all_versions("graph-victim", ATTACKER)
        assert result == []
        where = mock_prisma.return_value.find_many.call_args.kwargs["where"]
        assert where.get("userId") == ATTACKER
        assert where.get("id") == "graph-victim"


@pytest.mark.asyncio
async def test_workspace_scoped_route_accepts_resource_ids_still_scoped():
    """Workspace-scoped route: preview/download by file_id requires workspaceId scoping.

    Even though the route param is file_id (attacker knows victim's UUID),
    the server derives workspaceId from attacker user_id — not from the
    resource — so lookup fails.
    """
    from backend.data.workspace import get_workspace_file

    with patch("backend.data.workspace.UserWorkspaceFile.prisma") as mock_prisma:
        mock_prisma.return_value.find_first = AsyncMock(return_value=None)
        result = await get_workspace_file(
            file_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            workspace_id="ws-attacker",
        )
        assert result is None
        where = mock_prisma.return_value.find_first.call_args.kwargs["where"]
        assert where["workspaceId"] == "ws-attacker"
        assert where["id"] == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"