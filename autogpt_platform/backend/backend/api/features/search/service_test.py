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

    mocker.patch(
        "backend.data.workspace.get_workspace",
        new=AsyncMock(return_value=workspace),
    )
    if files is not None:
        mock_manager = MagicMock()
        mock_manager.list_files = AsyncMock(return_value=files)
        mocker.patch(
            "backend.util.workspace.WorkspaceManager", return_value=mock_manager
        )

    mocker.patch(
        "backend.copilot.model.get_user_sessions",
        new=AsyncMock(return_value=(list(sessions), len(sessions))),
    )
    return mock_lib


# ============================================================================
# global_search — hybrid branch (non-empty query)
# ============================================================================


@pytest.mark.asyncio
async def test_global_search_buckets_results_by_content_type():
    """One hybrid call per bucket; rows are sorted into the right list
    via their ``content_type`` field."""

    def side_effect(*, content_types, **_):
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
        if "WORKSPACE_FILE" in type_set:
            return (
                [
                    {
                        "content_id": "file-1",
                        "content_type": ContentType.WORKSPACE_FILE,
                        "metadata": {"name": "report.pdf", "mime_type": "pdf"},
                        "combined_score": 0.6,
                        "updated_at": datetime.datetime(2024, 1, 3),
                    }
                ],
                1,
            )
        if "CHAT_SESSION" in type_set:
            return (
                [
                    {
                        "content_id": "chat-1",
                        "content_type": ContentType.CHAT_SESSION,
                        "metadata": {"title": "My Chat"},
                        "combined_score": 0.5,
                        "updated_at": datetime.datetime(2024, 1, 4),
                    }
                ],
                1,
            )
        return ([], 0)

    mock_shim = MagicMock()
    mock_shim.unified_hybrid_search = AsyncMock(side_effect=side_effect)
    with patch("backend.api.features.search.service.search", return_value=mock_shim):
        result = await service.global_search(
            query="anything", user_id="u1", per_type_limit=4
        )

    # 3 parallel calls — one per bucket
    assert mock_shim.unified_hybrid_search.await_count == 3

    # Buckets carry the right rows
    assert [a.id for a in result.agents] == ["agent-1", "store-1"]
    assert [a.type for a in result.agents] == ["library_agent", "store_agent"]
    assert [f.id for f in result.files] == ["file-1"]
    assert result.files[0].type == "workspace_file"
    assert [c.id for c in result.chats] == ["chat-1"]
    assert result.chats[0].type == "chat_session"


@pytest.mark.asyncio
async def test_global_search_respects_per_type_limit():
    """Bucket cap is enforced even when the hybrid call returns more rows."""
    rows = [
        {
            "content_id": f"file-{i}",
            "content_type": ContentType.WORKSPACE_FILE,
            "metadata": {"name": f"file-{i}"},
            "combined_score": 0.5,
        }
        for i in range(10)
    ]

    def side_effect(*, content_types, **_):
        if any(ct.value == "WORKSPACE_FILE" for ct in content_types):
            return (rows, 10)
        return ([], 0)

    mock_shim = MagicMock()
    mock_shim.unified_hybrid_search = AsyncMock(side_effect=side_effect)
    with patch("backend.api.features.search.service.search", return_value=mock_shim):
        result = await service.global_search(query="x", user_id="u1", per_type_limit=3)

    assert len(result.files) == 3

    # The cap is also forwarded to the underlying call as page_size
    file_call = next(
        c
        for c in mock_shim.unified_hybrid_search.await_args_list
        if any(ct.value == "WORKSPACE_FILE" for ct in c.kwargs["content_types"])
    )
    assert file_call.kwargs["page_size"] == 3


@pytest.mark.asyncio
async def test_global_search_bucket_failure_does_not_kill_other_buckets():
    """If one bucket raises, the others still return their results."""

    async def side_effect(*, content_types, **_):
        if any(ct.value == "WORKSPACE_FILE" for ct in content_types):
            raise RuntimeError("simulated DB failure")
        return (
            (
                [
                    {
                        "content_id": "chat-1",
                        "content_type": ContentType.CHAT_SESSION,
                        "metadata": {"title": "Hi"},
                        "combined_score": 0.5,
                    }
                ]
                if any(ct.value == "CHAT_SESSION" for ct in content_types)
                else []
            ),
            1,
        )

    mock_shim = MagicMock()
    mock_shim.unified_hybrid_search = AsyncMock(side_effect=side_effect)
    with patch("backend.api.features.search.service.search", return_value=mock_shim):
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
