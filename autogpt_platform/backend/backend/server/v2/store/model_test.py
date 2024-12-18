import datetime

import prisma.enums

import backend.server.v2.store.model


def test_pagination():
    pagination = backend.server.v2.store.model.Pagination(
        total_items=100, total_pages=5, current_page=2, page_size=20
    )
    assert pagination.total_items == 100
    assert pagination.total_pages == 5
    assert pagination.current_page == 2
    assert pagination.page_size == 20


def test_store_agent():
    agent = backend.server.v2.store.model.StoreAgent(
        slug="test-agent",
        agent_name="Test Agent",
        agent_image="test.jpg",
        creator="creator1",
        creator_avatar="avatar.jpg",
        sub_heading="Test subheading",
        description="Test description",
        runs=50,
        rating=4.5,
    )
    assert agent.slug == "test-agent"
    assert agent.agent_name == "Test Agent"
    assert agent.runs == 50
    assert agent.rating == 4.5


def test_store_agents_response():
    response = backend.server.v2.store.model.StoreAgentsResponse(
        agents=[
            backend.server.v2.store.model.StoreAgent(
                slug="test-agent",
                agent_name="Test Agent",
                agent_image="test.jpg",
                creator="creator1",
                creator_avatar="avatar.jpg",
                sub_heading="Test subheading",
                description="Test description",
                runs=50,
                rating=4.5,
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            total_items=1, total_pages=1, current_page=1, page_size=20
        ),
    )
    assert len(response.agents) == 1
    assert response.pagination.total_items == 1


def test_store_agent_details():
    details = backend.server.v2.store.model.StoreAgentDetails(
        store_listing_version_id="version123",
        slug="test-agent",
        agent_name="Test Agent",
        agent_video="video.mp4",
        agent_image=["image1.jpg", "image2.jpg"],
        creator="creator1",
        creator_avatar="avatar.jpg",
        sub_heading="Test subheading",
        description="Test description",
        categories=["cat1", "cat2"],
        runs=50,
        rating=4.5,
        versions=["1.0", "2.0"],
        last_updated=datetime.datetime.now(),
    )
    assert details.slug == "test-agent"
    assert len(details.agent_image) == 2
    assert len(details.categories) == 2
    assert len(details.versions) == 2


def test_creator():
    creator = backend.server.v2.store.model.Creator(
        agent_rating=4.8,
        agent_runs=1000,
        name="Test Creator",
        username="creator1",
        description="Test description",
        avatar_url="avatar.jpg",
        num_agents=5,
        is_featured=False,
    )
    assert creator.name == "Test Creator"
    assert creator.num_agents == 5


def test_creators_response():
    response = backend.server.v2.store.model.CreatorsResponse(
        creators=[
            backend.server.v2.store.model.Creator(
                agent_rating=4.8,
                agent_runs=1000,
                name="Test Creator",
                username="creator1",
                description="Test description",
                avatar_url="avatar.jpg",
                num_agents=5,
                is_featured=False,
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            total_items=1, total_pages=1, current_page=1, page_size=20
        ),
    )
    assert len(response.creators) == 1
    assert response.pagination.total_items == 1


def test_creator_details():
    details = backend.server.v2.store.model.CreatorDetails(
        name="Test Creator",
        username="creator1",
        description="Test description",
        links=["link1.com", "link2.com"],
        avatar_url="avatar.jpg",
        agent_rating=4.8,
        agent_runs=1000,
        top_categories=["cat1", "cat2"],
    )
    assert details.name == "Test Creator"
    assert len(details.links) == 2
    assert details.agent_rating == 4.8
    assert len(details.top_categories) == 2


def test_store_submission():
    submission = backend.server.v2.store.model.StoreSubmission(
        agent_id="agent123",
        agent_version=1,
        sub_heading="Test subheading",
        name="Test Agent",
        slug="test-agent",
        description="Test description",
        image_urls=["image1.jpg", "image2.jpg"],
        date_submitted=datetime.datetime(2023, 1, 1),
        status=prisma.enums.SubmissionStatus.PENDING,
        runs=50,
        rating=4.5,
    )
    assert submission.name == "Test Agent"
    assert len(submission.image_urls) == 2
    assert submission.status == prisma.enums.SubmissionStatus.PENDING


def test_store_submissions_response():
    response = backend.server.v2.store.model.StoreSubmissionsResponse(
        submissions=[
            backend.server.v2.store.model.StoreSubmission(
                agent_id="agent123",
                agent_version=1,
                sub_heading="Test subheading",
                name="Test Agent",
                slug="test-agent",
                description="Test description",
                image_urls=["image1.jpg"],
                date_submitted=datetime.datetime(2023, 1, 1),
                status=prisma.enums.SubmissionStatus.PENDING,
                runs=50,
                rating=4.5,
            )
        ],
        pagination=backend.server.v2.store.model.Pagination(
            total_items=1, total_pages=1, current_page=1, page_size=20
        ),
    )
    assert len(response.submissions) == 1
    assert response.pagination.total_items == 1


def test_store_submission_request():
    request = backend.server.v2.store.model.StoreSubmissionRequest(
        agent_id="agent123",
        agent_version=1,
        slug="test-agent",
        name="Test Agent",
        sub_heading="Test subheading",
        video_url="video.mp4",
        image_urls=["image1.jpg", "image2.jpg"],
        description="Test description",
        categories=["cat1", "cat2"],
    )
    assert request.agent_id == "agent123"
    assert request.agent_version == 1
    assert len(request.image_urls) == 2
    assert len(request.categories) == 2
