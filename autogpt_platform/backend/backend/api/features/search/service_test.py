"""Unit tests for the unified search service."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ContentType

from backend.api.features.search import service
from backend.api.features.search.model import GlobalSearchResponse


def _patch_hybrid(return_value):
    """Patch the db_accessors.search() shim used by service.global_search."""
    mock_shim = MagicMock()
    mock_shim.unified_hybrid_search = AsyncMock(return_value=return_value)
    return (
        patch(
            "backend.api.features.search.service.search",
            return_value=mock_shim,
        ),
        mock_shim,
    )


# ============================================================================
# Fixtures + fakes
# ============================================================================


def _fake_library_agent(agent_id: str, name: str = "A"):
    from backend.api.features.library import model as library_model
    from backend.util.models import Pagination

    agent = library_model.LibraryAgent.model_construct(
        id=agent_id,
        graph_id=agent_id,
        graph_version=1,
        name=name,
        description="d",
        image_url=None,
        creator_name="c",
        creator_image_url="",
        input_schema={},
        output_schema={},
        credentials_input_schema=None,
        has_external_trigger=False,
        has_human_in_the_loop=False,
        has_sensitive_action=False,
        status=library_model.LibraryAgentStatus.COMPLETED,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
        created_at=datetime.datetime(2024, 1, 1),
        updated_at=datetime.datetime(2024, 1, 1),
    )
    return library_model.LibraryAgentResponse(
        agents=[agent],
        pagination=Pagination(
            total_items=1, total_pages=1, current_page=1, page_size=3
        ),
    )


def _fake_workspace_file(file_id: str = "f1"):
    from backend.data.workspace import WorkspaceFile

    return WorkspaceFile(
        id=file_id,
        workspace_id="ws-1",
        created_at=datetime.datetime(2024, 1, 1),
        updated_at=datetime.datetime(2024, 1, 2),
        name="report.pdf",
        path="/documents/report.pdf",
        storage_path="ws-1/documents/report.pdf",
        mime_type="application/pdf",
        size_bytes=123,
    )


def _fake_chat_session(session_id: str = "s1", title: str | None = "Title"):
    from backend.copilot.model import ChatSessionInfo

    return ChatSessionInfo(
        session_id=session_id,
        user_id="u1",
        title=title,
        usage=[],
        started_at=datetime.datetime(2024, 1, 1),
        updated_at=datetime.datetime(2024, 1, 5),
    )


@pytest.fixture(autouse=True)
def _clear_recent_cache():
    """The empty-query branch is decorated with ``@cached`` — clear the
    cache between tests so stale entries don't leak."""
    service._cached_recent_buckets.cache_clear()
    yield
    service._cached_recent_buckets.cache_clear()


def _mock_recent_sources(
    mocker,
    *,
    library_response=None,
    workspace=None,
    files=None,
    sessions=(),
):
    """Patch the three recent-bucket sources with sensible defaults.

    Returns the library_db mock so tests can assert on its call count.
    """
    mock_lib = MagicMock()
    mock_lib.list_library_agents = AsyncMock(
        return_value=library_response or _fake_library_agent("a-1")
    )
    mocker.patch(
        "backend.api.features.search.service.library_db", return_value=mock_lib
    )

    # Mock at the *use* site (where ``service`` rebinds the imported
    # symbol) rather than the *source* site. Service now imports these
    # at module top level (per ``AGENTS.md``), so patching the source
    # module no longer affects the already-bound name inside service.
    mocker.patch(
        "backend.api.features.search.service.get_workspace",
        new=AsyncMock(return_value=workspace),
    )
    if files is not None:
        mock_manager = MagicMock()
        mock_manager.list_files = AsyncMock(return_value=files)
        mocker.patch(
            "backend.api.features.search.service.WorkspaceManager",
            return_value=mock_manager,
        )

    mocker.patch(
        "backend.api.features.search.service.get_user_sessions",
        new=AsyncMock(return_value=(list(sessions), len(sessions))),
    )
    return mock_lib


# ============================================================================
# global_search — hybrid branch (non-empty query)
# ============================================================================


@pytest.mark.asyncio
async def test_global_search_buckets_results_by_content_type(mocker):
    """One hybrid call for agents; files & chats use direct DB queries —
    rows are sorted into the right bucket regardless of the source path."""

    def hybrid_side_effect(*, content_types, **_):
        type_set = {ct.value for ct in content_types}
        if "LIBRARY_AGENT" in type_set or "STORE_AGENT" in type_set:
            return (
                [
                    {
                        "content_id": "agent-1",
                        "content_type": ContentType.LIBRARY_AGENT,
                        "metadata": {"name": "My Agent"},
                        "combined_score": 0.8,
                        "updated_at": datetime.datetime(2024, 1, 1),
                    },
                    {
                        "content_id": "store-1",
                        "content_type": ContentType.STORE_AGENT,
                        "metadata": {"name": "Store Agent", "categories": ["ai"]},
                        "combined_score": 0.7,
                        "updated_at": datetime.datetime(2024, 1, 2),
                    },
                ],
                2,
            )
        return ([], 0)

    _mock_recent_sources(
        mocker,
        workspace=MagicMock(id="ws-1"),
        files=[_fake_workspace_file("file-1")],
        sessions=[_fake_chat_session("chat-1", title="My Chat")],
    )

    mock_shim = MagicMock()
    mock_shim.unified_hybrid_search = AsyncMock(side_effect=hybrid_side_effect)
    with patch("backend.api.features.search.service.search", return_value=mock_shim):
        result = await service.global_search(
            query="anything", user_id="u1", per_type_limit=4
        )

    # Only the agents bucket still goes through unified_hybrid_search;
    # files and chats now route through direct DB queries (see
    # _files_bucket / _chats_bucket).
    assert mock_shim.unified_hybrid_search.await_count == 1

    assert [a.id for a in result.agents] == ["agent-1", "store-1"]
    assert [a.type for a in result.agents] == ["library_agent", "store_agent"]
    assert [f.id for f in result.files] == ["file-1"]
    assert result.files[0].type == "workspace_file"
    assert [c.id for c in result.chats] == ["chat-1"]
    assert result.chats[0].type == "chat_session"


@pytest.mark.asyncio
async def test_global_search_respects_per_type_limit(mocker):
    """Bucket cap (``per_type_limit``) trims the rendered response after
    in-Python relevance reranking. The DB fetch limit is the larger of
    the bucket cap and ``_RELEVANCE_OVERFETCH_CAP`` (32) so the rerank
    has enough candidates to do meaningful work — see the docstring on
    ``_files_bucket``."""
    files = [_fake_workspace_file(f"file-{i}") for i in range(10)]

    mock_manager = MagicMock()
    mock_manager.list_files = AsyncMock(return_value=files)
    mocker.patch(
        "backend.api.features.search.service.WorkspaceManager",
        return_value=mock_manager,
    )
    mocker.patch(
        "backend.api.features.search.service.get_workspace",
        new=AsyncMock(return_value=MagicMock(id="ws-1")),
    )

    # Agents/chats are out of scope here — stub them to empty so the
    # gather() in global_search resolves cleanly.
    patcher, _ = _patch_hybrid(([], 0))
    mocker.patch(
        "backend.api.features.search.service.get_user_sessions",
        new=AsyncMock(return_value=([], 0)),
    )

    with patcher:
        result = await service.global_search(query="x", user_id="u1", per_type_limit=3)

    # Rendered response is trimmed to per_type_limit even though the
    # bucket fetched more for reranking.
    assert len(result.files) == 3
    # The DB fetch uses the relevance overfetch cap, NOT the bucket
    # cap — anything else would mean we're ranking too small a sample.
    assert (
        mock_manager.list_files.await_args.kwargs["limit"]
        == service._RELEVANCE_OVERFETCH_CAP
    )


@pytest.mark.asyncio
async def test_global_search_bucket_failure_does_not_kill_other_buckets(mocker):
    """If the files DB query raises, the chats & agents buckets still
    return their results — fan-out failures must not 500 the response."""

    mocker.patch(
        "backend.api.features.search.service.get_workspace",
        new=AsyncMock(side_effect=RuntimeError("simulated DB failure")),
    )
    mocker.patch(
        "backend.api.features.search.service.get_user_sessions",
        new=AsyncMock(return_value=([_fake_chat_session("chat-1", title="Hi")], 1)),
    )

    patcher, _ = _patch_hybrid(([], 0))
    with patcher:
        result = await service.global_search(query="x", user_id="u1")

    assert result.files == []  # failed bucket -> empty, not 500
    assert [c.id for c in result.chats] == ["chat-1"]


@pytest.mark.asyncio
async def test_global_search_forwards_user_id_to_every_bucket():
    patcher, mock_shim = _patch_hybrid(([], 0))
    with patcher:
        await service.global_search(query="x", user_id="user-42")

    for call in mock_shim.unified_hybrid_search.await_args_list:
        assert call.kwargs["user_id"] == "user-42"


# ============================================================================
# global_search — recent branch (empty query)
# ============================================================================


@pytest.mark.asyncio
async def test_global_search_empty_query_returns_recent_buckets(mocker):
    """Empty/whitespace ``q`` short-circuits the hybrid path and returns
    the most-recently-updated items per bucket."""
    _mock_recent_sources(
        mocker,
        workspace=MagicMock(id="ws-1"),
        files=[_fake_workspace_file()],
        sessions=[_fake_chat_session()],
    )

    # Patch the hybrid shim too so we can assert it's NOT invoked.
    patcher, mock_shim = _patch_hybrid(([], 0))
    with patcher:
        result = await service.global_search(query="   ", user_id="u1")

    assert isinstance(result, GlobalSearchResponse)
    assert [a.id for a in result.agents] == ["a-1"]
    assert result.agents[0].type == "library_agent"
    assert [f.id for f in result.files] == ["f1"]
    assert result.files[0].type == "workspace_file"
    assert [c.id for c in result.chats] == ["s1"]
    assert result.chats[0].type == "chat_session"

    mock_shim.unified_hybrid_search.assert_not_called()


@pytest.mark.asyncio
async def test_global_search_empty_query_caches_per_user(mocker):
    """Two empty-query calls with the same args reuse the cached response."""
    mock_lib = _mock_recent_sources(mocker)

    await service.global_search(query="", user_id="u1", per_type_limit=3)
    await service.global_search(query="", user_id="u1", per_type_limit=3)

    # Underlying list_library_agents was hit once — second call was cached.
    assert mock_lib.list_library_agents.await_count == 1


@pytest.mark.asyncio
async def test_global_search_empty_query_handles_user_without_workspace(mocker):
    """A brand-new user with no workspace row returns empty files without
    creating a workspace as a side-effect."""
    _mock_recent_sources(mocker)  # workspace=None by default

    result = await service.global_search(query="", user_id="u1")
    assert result.files == []


@pytest.mark.asyncio
async def test_global_search_empty_query_untitled_chat_falls_back(mocker):
    _mock_recent_sources(mocker, sessions=[_fake_chat_session(title=None)])

    result = await service.global_search(query="", user_id="u1")
    assert result.chats[0].title == "Untitled chat"


def test_db_accessor_search_resolves_unified_hybrid_search():
    """Regression: ``db_accessors.search()`` must expose
    ``unified_hybrid_search`` on the in-process branch.

    The unit tests above patch ``service.search`` so a wrong import in
    the accessor wouldn't surface. This guards against a repeat of the
    bug where the accessor was still pointing at
    ``backend.api.features.store.hybrid_search`` after the engine moved
    to ``backend.api.features.search.hybrid_search``.
    """
    from backend.data import db, db_accessors

    with patch.object(db, "is_connected", return_value=True):
        shim = db_accessors.search()
        assert callable(getattr(shim, "unified_hybrid_search", None)), (
            "db_accessors.search() must expose unified_hybrid_search; "
            "service.global_search hits this attribute on every non-empty "
            "query in production."
        )


# ============================================================================
# Relevance ordering — non-empty query reranks by literal name match
# ============================================================================


def _fake_workspace_file_named(file_id: str, name: str, created_at: datetime.datetime):
    from backend.data.workspace import WorkspaceFile

    return WorkspaceFile(
        id=file_id,
        workspace_id="ws-1",
        created_at=created_at,
        updated_at=created_at,
        name=name,
        path=f"/documents/{name}",
        storage_path=f"ws-1/documents/{name}",
        mime_type="application/pdf",
        size_bytes=1,
    )


def _fake_chat_session_named(
    session_id: str, title: str, updated_at: datetime.datetime
):
    from backend.copilot.model import ChatSessionInfo

    return ChatSessionInfo(
        session_id=session_id,
        user_id="u1",
        title=title,
        usage=[],
        started_at=updated_at,
        updated_at=updated_at,
    )


def test_title_relevance_score_ranks_exact_above_prefix_above_contains():
    """The three-tier literal-match score is the single knob behind the
    files/chats rerank — assert each rung individually so a future
    refactor can't silently flatten the scale."""
    assert service._title_relevance_score("Quarterly Report", "quarterly") == 2
    assert service._title_relevance_score("My Quarterly", "quarterly") == 1
    assert service._title_relevance_score("Quarterly", "quarterly") == 3
    assert service._title_relevance_score("Annual", "quarterly") == 0


@pytest.mark.asyncio
async def test_files_bucket_reranks_by_relevance_not_freshness(mocker):
    """Repro of the blocker on service.py:326 — a stale exact / prefix
    match must beat fresher contains-only matches within ``limit``."""
    # Freshness DESC from list_files: newest first
    files = [
        _fake_workspace_file_named(
            "f-new-1", "final-report-v3.pdf", datetime.datetime(2026, 5, 28)
        ),
        _fake_workspace_file_named(
            "f-new-2", "report-2026-Q1.pdf", datetime.datetime(2026, 5, 27)
        ),
        _fake_workspace_file_named(
            "f-new-3", "monthly-report.xlsx", datetime.datetime(2026, 5, 26)
        ),
        _fake_workspace_file_named(
            "f-new-4", "weekly-report.md", datetime.datetime(2026, 5, 25)
        ),
        # Older, but the literal best match (startswith)
        _fake_workspace_file_named(
            "f-old", "quarterly-report-overview.md", datetime.datetime(2026, 5, 20)
        ),
    ]
    mock_manager = MagicMock()
    mock_manager.list_files = AsyncMock(return_value=files)
    mocker.patch(
        "backend.api.features.search.service.WorkspaceManager",
        return_value=mock_manager,
    )
    mocker.patch(
        "backend.api.features.search.service.get_workspace",
        new=AsyncMock(return_value=MagicMock(id="ws-1")),
    )

    result = await service._files_bucket("u1", limit=1, query="quarterly")

    # The stale startswith match must outrank the fresh contains matches.
    assert [f.id for f in result] == ["f-old"]
    # Over-fetch must request at least _RELEVANCE_OVERFETCH_CAP candidates
    # so the rerank has enough headroom.
    assert (
        mock_manager.list_files.await_args.kwargs["limit"]
        >= service._RELEVANCE_OVERFETCH_CAP
    )


@pytest.mark.asyncio
async def test_chats_bucket_reranks_by_relevance_not_recency(mocker):
    """Same rerank guarantee for ``_chats_bucket`` — an older exact-title
    chat must beat fresher contains-only matches."""
    sessions = [
        _fake_chat_session_named(
            "s-new", "Reporting on Q2", datetime.datetime(2026, 5, 28)
        ),
        _fake_chat_session_named("s-old", "report", datetime.datetime(2026, 5, 20)),
    ]
    mocker.patch(
        "backend.api.features.search.service.get_user_sessions",
        new=AsyncMock(return_value=(sessions, len(sessions))),
    )

    result = await service._chats_bucket("u1", limit=1, query="report")

    # Exact case-insensitive match wins despite being older.
    assert [c.id for c in result] == ["s-old"]
