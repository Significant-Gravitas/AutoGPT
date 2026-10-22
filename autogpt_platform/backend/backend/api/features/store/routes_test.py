import datetime
import json
import uuid

import fastapi
import fastapi.testclient
import prisma.enums
import pytest
import pytest_mock
from autogpt_libs.auth import get_optional_user_id
from pydantic import BaseModel
from pytest_snapshot.plugin import Snapshot

from backend.api.features.store.db import StoreAgentsSortOptions
from backend.data.graph import GraphModelWithoutNodes

from . import model as store_model
from . import routes as store_routes

# Using a fixed timestamp for reproducible tests
# 2023 date is intentionally used to ensure tests work regardless of current year
FIXED_NOW = datetime.datetime(2023, 1, 1, 0, 0, 0)

app = fastapi.FastAPI()
app.include_router(store_routes.router)

client = fastapi.testclient.TestClient(app)

TEST_ORG_ID = "test-org-id"


def _test_request_context():
    from autogpt_libs.auth.models import RequestContext

    return RequestContext(
        user_id="test-user-id",
        org_id=TEST_ORG_ID,
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module.

    ``get_request_context`` is overridden with a deterministic context —
    the real dependency resolves the personal org from the database, which
    would make the ``organization_id`` forwarded to the db layer depend on
    test-database state.
    """
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    store_routes.store_cache.clear_all_caches()
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = _test_request_context
    yield
    app.dependency_overrides.clear()
    store_routes.store_cache.clear_all_caches()


def test_get_agents_defaults(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.StoreAgentsResponse(
        agents=[],
        pagination=store_model.Pagination(
            current_page=0,
            total_items=0,
            total_pages=0,
            page_size=10,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents")
    assert response.status_code == 200

    data = store_model.StoreAgentsResponse.model_validate(response.json())
    assert data.pagination.total_pages == 0
    assert data.agents == []

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "def_agts")
    mock_db_call.assert_called_once_with(
        featured=False,
        creators=None,
        sorted_by=None,
        search_query=None,
        category=None,
        page=1,
        page_size=20,
        public_search=True,
    )


def test_get_agents_featured(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.StoreAgentsResponse(
        agents=[
            store_model.StoreAgent(
                slug="featured-agent",
                agent_name="Featured Agent",
                agent_image="featured.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading="Featured agent subheading",
                description="Featured agent description",
                runs=100,
                rating=4.5,
                agent_graph_id="test-graph-1",
            )
        ],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?featured=true")
    assert response.status_code == 200
    data = store_model.StoreAgentsResponse.model_validate(response.json())
    assert len(data.agents) == 1
    assert data.agents[0].slug == "featured-agent"
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "feat_agts")
    mock_db_call.assert_called_once_with(
        featured=True,
        creators=None,
        sorted_by=None,
        search_query=None,
        category=None,
        page=1,
        page_size=20,
        public_search=True,
    )


def test_get_agents_by_creator(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.StoreAgentsResponse(
        agents=[
            store_model.StoreAgent(
                slug="creator-agent",
                agent_name="Creator Agent",
                agent_image="agent.jpg",
                creator="specific-creator",
                creator_avatar="avatar.jpg",
                sub_heading="Creator agent subheading",
                description="Creator agent description",
                runs=50,
                rating=4.0,
                agent_graph_id="test-graph-2",
            )
        ],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?creator=specific-creator")
    assert response.status_code == 200
    data = store_model.StoreAgentsResponse.model_validate(response.json())
    assert len(data.agents) == 1
    assert data.agents[0].creator == "specific-creator"
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "agts_by_creator")
    mock_db_call.assert_called_once_with(
        featured=False,
        creators=["specific-creator"],
        sorted_by=None,
        search_query=None,
        category=None,
        page=1,
        page_size=20,
        public_search=True,
    )


def test_get_agents_sorted(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.StoreAgentsResponse(
        agents=[
            store_model.StoreAgent(
                slug="top-agent",
                agent_name="Top Agent",
                agent_image="top.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading="Top agent subheading",
                description="Top agent description",
                runs=1000,
                rating=5.0,
                agent_graph_id="test-graph-3",
            )
        ],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?sorted_by=runs")
    assert response.status_code == 200
    data = store_model.StoreAgentsResponse.model_validate(response.json())
    assert len(data.agents) == 1
    assert data.agents[0].runs == 1000
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "agts_sorted")
    mock_db_call.assert_called_once_with(
        featured=False,
        creators=None,
        sorted_by=StoreAgentsSortOptions.RUNS,
        search_query=None,
        category=None,
        page=1,
        page_size=20,
        public_search=True,
    )


def test_get_agents_search(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.StoreAgentsResponse(
        agents=[
            store_model.StoreAgent(
                slug="search-agent",
                agent_name="Search Agent",
                agent_image="search.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading="Search agent subheading",
                description="Specific search term description",
                runs=75,
                rating=4.2,
                agent_graph_id="test-graph-search",
            )
        ],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?search_query=specific")
    assert response.status_code == 200
    data = store_model.StoreAgentsResponse.model_validate(response.json())
    assert len(data.agents) == 1
    assert "specific" in data.agents[0].description.lower()
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "agts_search")
    mock_db_call.assert_called_once_with(
        featured=False,
        creators=None,
        sorted_by=None,
        search_query="specific",
        category=None,
        page=1,
        page_size=20,
        public_search=True,
    )


def test_get_agents_category(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.StoreAgentsResponse(
        agents=[
            store_model.StoreAgent(
                slug="category-agent",
                agent_name="Category Agent",
                agent_image="category.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading="Category agent subheading",
                description="Category agent description",
                runs=60,
                rating=4.1,
                agent_graph_id="test-graph-category",
            )
        ],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?category=test-category")
    assert response.status_code == 200
    data = store_model.StoreAgentsResponse.model_validate(response.json())
    assert len(data.agents) == 1
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "agts_category")
    mock_db_call.assert_called_once_with(
        featured=False,
        creators=None,
        sorted_by=None,
        search_query=None,
        category="test-category",
        page=1,
        page_size=20,
        public_search=True,
    )


def test_get_agents_pagination(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.StoreAgentsResponse(
        agents=[
            store_model.StoreAgent(
                slug=f"agent-{i}",
                agent_name=f"Agent {i}",
                agent_image=f"agent{i}.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading=f"Agent {i} subheading",
                description=f"Agent {i} description",
                runs=i * 10,
                rating=4.0,
                agent_graph_id="test-graph-2",
            )
            for i in range(5)
        ],
        pagination=store_model.Pagination(
            current_page=2,
            total_items=15,
            total_pages=3,
            page_size=5,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?page=2&page_size=5")
    assert response.status_code == 200
    data = store_model.StoreAgentsResponse.model_validate(response.json())
    assert len(data.agents) == 5
    assert data.pagination.current_page == 2
    assert data.pagination.page_size == 5
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "agts_pagination")
    mock_db_call.assert_called_once_with(
        featured=False,
        creators=None,
        sorted_by=None,
        search_query=None,
        category=None,
        page=2,
        page_size=5,
        public_search=True,
    )


def test_get_agents_malformed_request(mocker: pytest_mock.MockFixture):
    # Test with invalid page number
    response = client.get("/agents?page=-1")
    assert response.status_code == 422

    # Test with invalid page size
    response = client.get("/agents?page_size=0")
    assert response.status_code == 422

    # Test with non-numeric values
    response = client.get("/agents?page=abc&page_size=def")
    assert response.status_code == 422

    # Verify no DB calls were made
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agents")
    mock_db_call.assert_not_called()


@pytest.fixture
def one_search_a_minute(monkeypatch):
    from backend.api.features.search import rate_limit

    class _Frozen(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2026, 1, 1, 12, 0, 30, tzinfo=datetime.UTC)

    monkeypatch.setattr(
        rate_limit.settings.config, "public_search_requests_per_minute", 1
    )
    monkeypatch.setattr(rate_limit, "datetime", _Frozen)
    # A cold Redis connection can outlast the deadline, and the limit then fails open.
    monkeypatch.setattr(rate_limit, "PUBLIC_SEARCH_REDIS_TIMEOUT_SECONDS", 10)


@pytest.mark.parametrize(
    "path", ["/search?query=sheets", "/agents?search_query=sheets"]
)
def test_a_caller_without_the_frontends_token_is_held_to_the_search_limit(
    mocker: pytest_mock.MockFixture, one_search_a_minute, path: str
) -> None:
    _mock_search_backends(mocker)
    caller = _a_new_caller()

    first = caller.get(path, headers=_claimed_address("192.0.2.1"))
    blocked = caller.get(path, headers=_claimed_address("192.0.2.2"))

    assert first.status_code == 200
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "30"


@pytest.mark.parametrize(
    "path", ["/search?query=sheets", "/agents?search_query=sheets"]
)
def test_signed_in_searches_count_per_account_and_signed_out_ones_per_address(
    mocker: pytest_mock.MockFixture, one_search_a_minute, monkeypatch, path: str
) -> None:
    from backend.api.features.search import rate_limit

    _mock_search_backends(mocker)
    monkeypatch.setattr(
        rate_limit.settings.config, "public_search_user_requests_per_minute", 2
    )
    app.dependency_overrides[get_optional_user_id] = _user_from_test_header
    office = _a_new_caller()
    alice, bob = _signed_in(), _signed_in()

    alice_statuses = [office.get(path, headers=alice).status_code for _ in range(3)]
    bob_statuses = [office.get(path, headers=bob).status_code for _ in range(2)]
    signed_out_statuses = [office.get(path).status_code for _ in range(2)]

    assert alice_statuses == [200, 200, 429]
    assert bob_statuses == [200, 200]
    assert signed_out_statuses == [200, 429]


def test_listing_agents_without_a_search_is_not_limited(
    mocker: pytest_mock.MockFixture, one_search_a_minute
) -> None:
    _mock_search_backends(mocker)
    caller = _a_new_caller()

    statuses = [caller.get("/agents").status_code for _ in range(3)]

    assert statuses == [200, 200, 200]


def test_store_search_counts_against_the_public_embedding_budget(
    mocker: pytest_mock.MockFixture,
) -> None:
    engine = _mock_search_backends(mocker)

    response = _a_new_caller().get("/search?query=sheets")

    assert response.status_code == 200
    assert engine.await_args.kwargs["public_search"] is True


def _a_new_caller() -> fastapi.testclient.TestClient:
    """A client with a socket address of its own, so no test shares its window."""
    return fastapi.testclient.TestClient(app, client=(f"caller-{uuid.uuid4().hex}", 1))


def _signed_in() -> dict[str, str]:
    return {"X-Test-User": f"user-{uuid.uuid4().hex}"}


def _user_from_test_header(
    x_test_user: str | None = fastapi.Header(default=None),
) -> str | None:
    return x_test_user


def _claimed_address(ip: str) -> dict[str, str]:
    return {"CF-Connecting-IP": ip, "X-Forwarded-For": ip, "X-Real-IP": ip}


def _mock_search_backends(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.store.db.get_store_agents",
        return_value=store_model.StoreAgentsResponse(
            agents=[],
            pagination=store_model.Pagination(
                current_page=1, total_items=0, total_pages=0, page_size=20
            ),
        ),
    )
    return mocker.patch.object(
        store_routes.search_engine,
        "unified_hybrid_search",
        new_callable=mocker.AsyncMock,
        return_value=([], 0),
    )


def test_get_agent_details(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.StoreAgentDetails(
        store_listing_version_id="test-version-id",
        slug="test-agent",
        agent_name="Test Agent",
        agent_video="video.mp4",
        agent_output_demo="demo.mp4",
        agent_image=["image1.jpg", "image2.jpg"],
        creator="creator1",
        creator_avatar="avatar1.jpg",
        sub_heading="Test agent subheading",
        description="Test agent description",
        categories=["category1", "category2"],
        runs=100,
        rating=4.5,
        versions=["1.0.0", "1.1.0"],
        graph_versions=["1", "2"],
        graph_id="test-graph-id",
        last_updated=FIXED_NOW,
        active_version_id="test-version-id",
        has_approved_version=True,
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_agent_details")
    mock_db_call.return_value = mocked_value

    response = client.get("/agents/creator1/test-agent")
    assert response.status_code == 200

    data = store_model.StoreAgentDetails.model_validate(response.json())
    assert data.agent_name == "Test Agent"
    assert data.creator == "creator1"
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "agt_details")
    mock_db_call.assert_called_once_with(
        username="creator1", agent_name="test-agent", include_changelog=False
    )


def test_get_creators_defaults(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.CreatorsResponse(
        creators=[],
        pagination=store_model.Pagination(
            current_page=0,
            total_items=0,
            total_pages=0,
            page_size=10,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_creators")
    mock_db_call.return_value = mocked_value

    response = client.get("/creators")
    assert response.status_code == 200

    data = store_model.CreatorsResponse.model_validate(response.json())
    assert data.pagination.total_pages == 0
    assert data.creators == []
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "def_creators")
    mock_db_call.assert_called_once_with(
        featured=False, search_query=None, sorted_by=None, page=1, page_size=20
    )


def test_get_creators_pagination(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.CreatorsResponse(
        creators=[
            store_model.CreatorDetails(
                name=f"Creator {i}",
                username=f"creator{i}",
                avatar_url=f"avatar{i}.jpg",
                description=f"Creator {i} description",
                links=[f"user{i}.link.com"],
                is_featured=False,
                num_agents=1,
                agent_runs=100,
                agent_rating=4.5,
                top_categories=["cat1", "cat2", "cat3"],
            )
            for i in range(5)
        ],
        pagination=store_model.Pagination(
            current_page=2,
            total_items=15,
            total_pages=3,
            page_size=5,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_creators")
    mock_db_call.return_value = mocked_value

    response = client.get("/creators?page=2&page_size=5")
    assert response.status_code == 200

    data = store_model.CreatorsResponse.model_validate(response.json())
    assert len(data.creators) == 5
    assert data.pagination.current_page == 2
    assert data.pagination.page_size == 5
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "creators_pagination")
    mock_db_call.assert_called_once_with(
        featured=False, search_query=None, sorted_by=None, page=2, page_size=5
    )


def test_get_creators_malformed_request(mocker: pytest_mock.MockFixture):
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_creators")

    # Test with invalid page number
    response = client.get("/creators?page=-1")
    assert response.status_code == 422

    # Test with invalid page size
    response = client.get("/creators?page_size=0")
    assert response.status_code == 422
    response = client.get("/creators?page_size=101")
    assert response.status_code == 422

    # Test with non-numeric values
    response = client.get("/creators?page=abc&page_size=def")
    assert response.status_code == 422

    mock_db_call.assert_not_called()


def test_get_creator_details(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    mocked_value = store_model.CreatorDetails(
        name="Test User",
        username="creator1",
        avatar_url="avatar.jpg",
        description="Test creator description",
        links=["link1.com", "link2.com"],
        is_featured=True,
        num_agents=5,
        agent_runs=1000,
        agent_rating=4.8,
        top_categories=["category1", "category2"],
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_creator")
    mock_db_call.return_value = mocked_value

    response = client.get("/creators/creator1")
    assert response.status_code == 200

    data = store_model.CreatorDetails.model_validate(response.json())
    assert data.username == "creator1"
    assert data.name == "Test User"
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "creator_details")
    mock_db_call.assert_called_once_with(username="creator1")


def test_get_submissions_success(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    mocked_value = store_model.StoreSubmissionsResponse(
        submissions=[
            store_model.StoreSubmission(
                listing_id="test-listing-id",
                user_id="test-user-id",
                slug="test-agent",
                listing_version_id="test-version-id",
                listing_version=1,
                graph_id="test-agent-id",
                graph_version=1,
                name="Test Agent",
                sub_heading="Test agent subheading",
                description="Test agent description",
                instructions="Click the button!",
                categories=["test-category"],
                image_urls=["test.jpg"],
                video_url="test.mp4",
                agent_output_demo_url="demo_video.mp4",
                submitted_at=FIXED_NOW,
                changes_summary="Initial Submission",
                status=prisma.enums.SubmissionStatus.APPROVED,
                run_count=50,
                review_count=5,
                review_avg_rating=4.2,
            )
        ],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
        stats=store_model.SubmissionStats(
            total=1,
            approved=1,
            pending=0,
            total_runs=50,
            average_rating=4.2,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    mock_db_call.return_value = mocked_value

    response = client.get("/submissions")
    assert response.status_code == 200

    data = store_model.StoreSubmissionsResponse.model_validate(response.json())
    assert len(data.submissions) == 1
    assert data.submissions[0].name == "Test Agent"
    assert data.pagination.current_page == 1
    assert data.stats.total == 1
    assert data.stats.total_runs == 50
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "sub_success")
    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        organization_id=TEST_ORG_ID,
        page=1,
        page_size=20,
        search_query=None,
        statuses=None,
        sort_key=None,
        sort_dir="desc",
    )


def test_get_submissions_pagination(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    mocked_value = store_model.StoreSubmissionsResponse(
        submissions=[],
        pagination=store_model.Pagination(
            current_page=2,
            total_items=10,
            total_pages=2,
            page_size=5,
        ),
        stats=store_model.SubmissionStats(
            total=10,
            approved=4,
            pending=3,
            total_runs=12_345,
            average_rating=3.7,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    mock_db_call.return_value = mocked_value

    response = client.get("/submissions?page=2&page_size=5")
    assert response.status_code == 200

    data = store_model.StoreSubmissionsResponse.model_validate(response.json())
    assert data.pagination.current_page == 2
    assert data.pagination.page_size == 5
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(json.dumps(response.json(), indent=2), "sub_pagination")
    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        organization_id=TEST_ORG_ID,
        page=2,
        page_size=5,
        search_query=None,
        statuses=None,
        sort_key=None,
        sort_dir="desc",
    )


def test_get_submissions_forwards_search_query(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mocked_value = store_model.StoreSubmissionsResponse(
        submissions=[],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=0,
            total_pages=0,
            page_size=20,
        ),
        stats=store_model.SubmissionStats(
            total=0, approved=0, pending=0, total_runs=0, average_rating=None
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    mock_db_call.return_value = mocked_value

    response = client.get("/submissions?search_query=invoice%20agent")
    assert response.status_code == 200
    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        organization_id=TEST_ORG_ID,
        page=1,
        page_size=20,
        search_query="invoice agent",
        statuses=None,
        sort_key=None,
        sort_dir="desc",
    )


def test_get_submissions_forwards_statuses(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mocked_value = store_model.StoreSubmissionsResponse(
        submissions=[],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=0,
            total_pages=0,
            page_size=20,
        ),
        stats=store_model.SubmissionStats(
            total=0, approved=0, pending=0, total_runs=0, average_rating=None
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    mock_db_call.return_value = mocked_value

    response = client.get("/submissions?statuses=PENDING,APPROVED")
    assert response.status_code == 200
    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        organization_id=TEST_ORG_ID,
        page=1,
        page_size=20,
        search_query=None,
        statuses=[
            prisma.enums.SubmissionStatus.PENDING,
            prisma.enums.SubmissionStatus.APPROVED,
        ],
        sort_key=None,
        sort_dir="desc",
    )


def test_get_submissions_forwards_sort(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mocked_value = store_model.StoreSubmissionsResponse(
        submissions=[],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=0,
            total_pages=0,
            page_size=20,
        ),
        stats=store_model.SubmissionStats(
            total=0, approved=0, pending=0, total_runs=0, average_rating=None
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    mock_db_call.return_value = mocked_value

    response = client.get("/submissions?sort_key=runs&sort_dir=asc")
    assert response.status_code == 200
    mock_db_call.assert_called_once_with(
        user_id=test_user_id,
        organization_id=TEST_ORG_ID,
        page=1,
        page_size=20,
        search_query=None,
        statuses=None,
        sort_key="runs",
        sort_dir="asc",
    )


def test_get_submissions_rejects_invalid_sort(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    response = client.get("/submissions?sort_key=name")
    assert response.status_code == 422
    mock_db_call.assert_not_called()


def test_get_submissions_rejects_invalid_status(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    response = client.get("/submissions?statuses=INVALID")
    assert response.status_code == 422
    mock_db_call.assert_not_called()


def test_get_submissions_search_query_too_long(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    response = client.get("/submissions?search_query=" + "x" * 101)
    assert response.status_code == 422
    mock_db_call.assert_not_called()


def test_get_my_unpublished_agents_forwards_search_query(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mocked_value = store_model.MyUnpublishedAgentsResponse(
        agents=[],
        pagination=store_model.Pagination(
            current_page=1,
            total_items=0,
            total_pages=0,
            page_size=10,
        ),
    )
    mock_db_call = mocker.patch("backend.api.features.store.db.get_my_agents")
    mock_db_call.return_value = mocked_value

    response = client.get(
        "/my-unpublished-agents?page=1&page_size=10&search_query=scraper"
    )
    assert response.status_code == 200
    mock_db_call.assert_called_once_with(
        test_user_id,
        page=1,
        page_size=10,
        sort_by=store_model.MyAgentsSortBy.MOST_RECENT,
        search_query="scraper",
    )


def test_get_my_unpublished_agents_search_query_too_long(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_db_call = mocker.patch("backend.api.features.store.db.get_my_agents")
    response = client.get("/my-unpublished-agents?search_query=" + "x" * 101)
    assert response.status_code == 422
    mock_db_call.assert_not_called()


def test_get_submissions_malformed_request(mocker: pytest_mock.MockFixture):
    # Test with invalid page number
    response = client.get("/submissions?page=-1")
    assert response.status_code == 422

    # Test with invalid page size
    response = client.get("/submissions?page_size=0")
    assert response.status_code == 422

    # Test with non-numeric values
    response = client.get("/submissions?page=abc&page_size=def")
    assert response.status_code == 422

    # Verify no DB calls were made
    mock_db_call = mocker.patch("backend.api.features.store.db.get_store_submissions")
    mock_db_call.assert_not_called()


class _DownloadedGraph(BaseModel):
    id: str
    version: int


def test_listing_graph_outline_names_no_publisher_account(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Any signed-in user can read a listing's outline, so it carries no ids
    of the account or organization that published the workflow."""
    mocker.patch(
        "backend.api.features.store.db.get_available_graph",
        mocker.AsyncMock(
            return_value=GraphModelWithoutNodes(
                id="graph-1",
                version=1,
                name="Listed",
                description="",
                user_id="publisher",
                organization_id="publisher-org",
                team_id="publisher-team",
                created_at=FIXED_NOW,
            )
        ),
    )

    response = client.get("/listings/versions/slv-1/graph")

    assert response.status_code == 200
    outline = response.json()
    assert outline["id"] == "graph-1"
    assert (outline["user_id"], outline["organization_id"], outline["team_id"]) == (
        "",
        None,
        None,
    )


def test_download_agent_file_sends_listing_downloaded(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.features.store.db.get_agent",
        mocker.AsyncMock(return_value=_DownloadedGraph(id="graph-1", version=3)),
    )
    track = mocker.patch("backend.api.features.store.routes.track_listing_downloaded")
    app.dependency_overrides[get_optional_user_id] = lambda: "test-user-id"

    response = client.get("/listings/versions/slv-1/graph/download")

    assert response.status_code == 200
    track.assert_called_once_with(
        user_id="test-user-id", store_listing_version_id="slv-1", graph_id="graph-1"
    )


def test_signed_out_download_passes_no_user(mocker: pytest_mock.MockFixture) -> None:
    mocker.patch(
        "backend.api.features.store.db.get_agent",
        mocker.AsyncMock(return_value=_DownloadedGraph(id="graph-1", version=3)),
    )
    track = mocker.patch("backend.api.features.store.routes.track_listing_downloaded")

    response = client.get("/listings/versions/slv-1/graph/download")

    assert response.status_code == 200
    assert track.call_args.kwargs["user_id"] is None
