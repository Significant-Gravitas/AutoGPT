import datetime

import autogpt_libs.auth.depends
import autogpt_libs.auth.middleware
import fastapi
import fastapi.testclient
import prisma.enums
import pytest_mock

import backend.server.v2.store.model
import backend.server.v2.store.routes

app = fastapi.FastAPI()
app.include_router(backend.server.v2.store.routes.router)

client = fastapi.testclient.TestClient(app)


def override_auth_middleware():
    """Override auth middleware for testing"""
    return {"sub": "test-user-id"}


def override_get_user_id():
    """Override get_user_id for testing"""
    return "test-user-id"


app.dependency_overrides[autogpt_libs.auth.middleware.auth_middleware] = (
    override_auth_middleware
)
app.dependency_overrides[autogpt_libs.auth.depends.get_user_id] = override_get_user_id


def test_get_agents_defaults(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreAgentsResponse(
        agents=[],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=0,
            total_items=0,
            total_pages=0,
            page_size=10,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents")
    assert response.status_code == 200

    data = backend.server.v2.store.model.StoreAgentsResponse.model_validate(
        response.json()
    )
    assert data.pagination.total_pages == 0
    assert data.agents == []
    mock_db_call.assert_called_once_with(
        featured=False,
        creator=None,
        sorted_by=None,
        search_query=None,
        category=None,
        page=1,
        page_size=20,
    )


def test_get_agents_featured(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreAgentsResponse(
        agents=[
            backend.server.v2.store.model.StoreAgent(
                slug="featured-agent",
                agent_name="Featured Agent",
                agent_image="featured.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading="Featured agent subheading",
                description="Featured agent description",
                runs=100,
                rating=4.5,
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?featured=true")
    assert response.status_code == 200
    data = backend.server.v2.store.model.StoreAgentsResponse.model_validate(
        response.json()
    )
    assert len(data.agents) == 1
    assert data.agents[0].slug == "featured-agent"
    mock_db_call.assert_called_once_with(
        featured=True,
        creator=None,
        sorted_by=None,
        search_query=None,
        category=None,
        page=1,
        page_size=20,
    )


def test_get_agents_by_creator(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreAgentsResponse(
        agents=[
            backend.server.v2.store.model.StoreAgent(
                slug="creator-agent",
                agent_name="Creator Agent",
                agent_image="agent.jpg",
                creator="specific-creator",
                creator_avatar="avatar.jpg",
                sub_heading="Creator agent subheading",
                description="Creator agent description",
                runs=50,
                rating=4.0,
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?creator=specific-creator")
    assert response.status_code == 200
    data = backend.server.v2.store.model.StoreAgentsResponse.model_validate(
        response.json()
    )
    assert len(data.agents) == 1
    assert data.agents[0].creator == "specific-creator"
    mock_db_call.assert_called_once_with(
        featured=False,
        creator="specific-creator",
        sorted_by=None,
        search_query=None,
        category=None,
        page=1,
        page_size=20,
    )


def test_get_agents_sorted(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreAgentsResponse(
        agents=[
            backend.server.v2.store.model.StoreAgent(
                slug="top-agent",
                agent_name="Top Agent",
                agent_image="top.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading="Top agent subheading",
                description="Top agent description",
                runs=1000,
                rating=5.0,
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?sorted_by=runs")
    assert response.status_code == 200
    data = backend.server.v2.store.model.StoreAgentsResponse.model_validate(
        response.json()
    )
    assert len(data.agents) == 1
    assert data.agents[0].runs == 1000
    mock_db_call.assert_called_once_with(
        featured=False,
        creator=None,
        sorted_by="runs",
        search_query=None,
        category=None,
        page=1,
        page_size=20,
    )


def test_get_agents_search(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreAgentsResponse(
        agents=[
            backend.server.v2.store.model.StoreAgent(
                slug="search-agent",
                agent_name="Search Agent",
                agent_image="search.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading="Search agent subheading",
                description="Specific search term description",
                runs=75,
                rating=4.2,
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?search_query=specific")
    assert response.status_code == 200
    data = backend.server.v2.store.model.StoreAgentsResponse.model_validate(
        response.json()
    )
    assert len(data.agents) == 1
    assert "specific" in data.agents[0].description.lower()
    mock_db_call.assert_called_once_with(
        featured=False,
        creator=None,
        sorted_by=None,
        search_query="specific",
        category=None,
        page=1,
        page_size=20,
    )


def test_get_agents_category(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreAgentsResponse(
        agents=[
            backend.server.v2.store.model.StoreAgent(
                slug="category-agent",
                agent_name="Category Agent",
                agent_image="category.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading="Category agent subheading",
                description="Category agent description",
                runs=60,
                rating=4.1,
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?category=test-category")
    assert response.status_code == 200
    data = backend.server.v2.store.model.StoreAgentsResponse.model_validate(
        response.json()
    )
    assert len(data.agents) == 1
    mock_db_call.assert_called_once_with(
        featured=False,
        creator=None,
        sorted_by=None,
        search_query=None,
        category="test-category",
        page=1,
        page_size=20,
    )


def test_get_agents_pagination(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreAgentsResponse(
        agents=[
            backend.server.v2.store.model.StoreAgent(
                slug=f"agent-{i}",
                agent_name=f"Agent {i}",
                agent_image=f"agent{i}.jpg",
                creator="creator1",
                creator_avatar="avatar1.jpg",
                sub_heading=f"Agent {i} subheading",
                description=f"Agent {i} description",
                runs=i * 10,
                rating=4.0,
            )
            for i in range(5)
        ],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=2,
            total_items=15,
            total_pages=3,
            page_size=5,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agents")
    mock_db_call.return_value = mocked_value
    response = client.get("/agents?page=2&page_size=5")
    assert response.status_code == 200
    data = backend.server.v2.store.model.StoreAgentsResponse.model_validate(
        response.json()
    )
    assert len(data.agents) == 5
    assert data.pagination.current_page == 2
    assert data.pagination.page_size == 5
    mock_db_call.assert_called_once_with(
        featured=False,
        creator=None,
        sorted_by=None,
        search_query=None,
        category=None,
        page=2,
        page_size=5,
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
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agents")
    mock_db_call.assert_not_called()


def test_get_agent_details(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreAgentDetails(
        store_listing_version_id="test-version-id",
        slug="test-agent",
        agent_name="Test Agent",
        agent_video="video.mp4",
        agent_image=["image1.jpg", "image2.jpg"],
        creator="creator1",
        creator_avatar="avatar1.jpg",
        sub_heading="Test agent subheading",
        description="Test agent description",
        categories=["category1", "category2"],
        runs=100,
        rating=4.5,
        versions=["1.0.0", "1.1.0"],
        last_updated=datetime.datetime.now(),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_agent_details")
    mock_db_call.return_value = mocked_value

    response = client.get("/agents/creator1/test-agent")
    assert response.status_code == 200

    data = backend.server.v2.store.model.StoreAgentDetails.model_validate(
        response.json()
    )
    assert data.agent_name == "Test Agent"
    assert data.creator == "creator1"
    mock_db_call.assert_called_once_with(username="creator1", agent_name="test-agent")


def test_get_creators_defaults(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.CreatorsResponse(
        creators=[],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=0,
            total_items=0,
            total_pages=0,
            page_size=10,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_creators")
    mock_db_call.return_value = mocked_value

    response = client.get("/creators")
    assert response.status_code == 200

    data = backend.server.v2.store.model.CreatorsResponse.model_validate(
        response.json()
    )
    assert data.pagination.total_pages == 0
    assert data.creators == []
    mock_db_call.assert_called_once_with(
        featured=False, search_query=None, sorted_by=None, page=1, page_size=20
    )


def test_get_creators_pagination(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.CreatorsResponse(
        creators=[
            backend.server.v2.store.model.Creator(
                name=f"Creator {i}",
                username=f"creator{i}",
                description=f"Creator {i} description",
                avatar_url=f"avatar{i}.jpg",
                num_agents=1,
                agent_rating=4.5,
                agent_runs=100,
                is_featured=False,
            )
            for i in range(5)
        ],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=2,
            total_items=15,
            total_pages=3,
            page_size=5,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_creators")
    mock_db_call.return_value = mocked_value

    response = client.get("/creators?page=2&page_size=5")
    assert response.status_code == 200

    data = backend.server.v2.store.model.CreatorsResponse.model_validate(
        response.json()
    )
    assert len(data.creators) == 5
    assert data.pagination.current_page == 2
    assert data.pagination.page_size == 5
    mock_db_call.assert_called_once_with(
        featured=False, search_query=None, sorted_by=None, page=2, page_size=5
    )


def test_get_creators_malformed_request(mocker: pytest_mock.MockFixture):
    # Test with invalid page number
    response = client.get("/creators?page=-1")
    assert response.status_code == 422

    # Test with invalid page size
    response = client.get("/creators?page_size=0")
    assert response.status_code == 422

    # Test with non-numeric values
    response = client.get("/creators?page=abc&page_size=def")
    assert response.status_code == 422

    # Verify no DB calls were made
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_creators")
    mock_db_call.assert_not_called()


def test_get_creator_details(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.CreatorDetails(
        name="Test User",
        username="creator1",
        description="Test creator description",
        links=["link1.com", "link2.com"],
        avatar_url="avatar.jpg",
        agent_rating=4.8,
        agent_runs=1000,
        top_categories=["category1", "category2"],
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_creator_details")
    mock_db_call.return_value = mocked_value

    response = client.get("/creator/creator1")
    assert response.status_code == 200

    data = backend.server.v2.store.model.CreatorDetails.model_validate(response.json())
    assert data.username == "creator1"
    assert data.name == "Test User"
    mock_db_call.assert_called_once_with(username="creator1")


def test_get_submissions_success(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreSubmissionsResponse(
        submissions=[
            backend.server.v2.store.model.StoreSubmission(
                name="Test Agent",
                description="Test agent description",
                image_urls=["test.jpg"],
                date_submitted=datetime.datetime.now(),
                status=prisma.enums.SubmissionStatus.APPROVED,
                runs=50,
                rating=4.2,
                agent_id="test-agent-id",
                agent_version=1,
                sub_heading="Test agent subheading",
                slug="test-agent",
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=1,
            total_items=1,
            total_pages=1,
            page_size=20,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_submissions")
    mock_db_call.return_value = mocked_value

    response = client.get("/submissions")
    assert response.status_code == 200

    data = backend.server.v2.store.model.StoreSubmissionsResponse.model_validate(
        response.json()
    )
    assert len(data.submissions) == 1
    assert data.submissions[0].name == "Test Agent"
    assert data.pagination.current_page == 1
    mock_db_call.assert_called_once_with(user_id="test-user-id", page=1, page_size=20)


def test_get_submissions_pagination(mocker: pytest_mock.MockFixture):
    mocked_value = backend.server.v2.store.model.StoreSubmissionsResponse(
        submissions=[],
        pagination=backend.server.v2.store.model.Pagination(
            current_page=2,
            total_items=10,
            total_pages=2,
            page_size=5,
        ),
    )
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_submissions")
    mock_db_call.return_value = mocked_value

    response = client.get("/submissions?page=2&page_size=5")
    assert response.status_code == 200

    data = backend.server.v2.store.model.StoreSubmissionsResponse.model_validate(
        response.json()
    )
    assert data.pagination.current_page == 2
    assert data.pagination.page_size == 5
    mock_db_call.assert_called_once_with(user_id="test-user-id", page=2, page_size=5)


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
    mock_db_call = mocker.patch("backend.server.v2.store.db.get_store_submissions")
    mock_db_call.assert_not_called()
