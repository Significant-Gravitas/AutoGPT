import uuid
from datetime import UTC, datetime
from os import getenv

import pytest
from pydantic import SecretStr

from backend.blocks.firecrawl.scrape import FirecrawlScrapeBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.blocks.llm import AITextGeneratorBlock
from backend.data.db import prisma
from backend.data.graph import Graph, Link, Node, create_graph
from backend.data.model import APIKeyCredentials
from backend.data.user import get_or_create_user
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.server.v2.chat.model import ChatSession
from backend.server.v2.store import db as store_db


def make_session(user_id: str | None = None):
    return ChatSession(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        messages=[],
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        successful_agent_runs={},
        successful_agent_schedules={},
    )


@pytest.fixture(scope="session")
async def setup_test_data():
    """
    Set up test data for run_agent tests:
    1. Create a test user
    2. Create a test graph (agent input -> agent output)
    3. Create a store listing and store listing version
    4. Approve the store listing version
    """
    # 1. Create a test user
    user_data = {
        "sub": f"test-user-{uuid.uuid4()}",
        "email": f"test-{uuid.uuid4()}@example.com",
    }
    user = await get_or_create_user(user_data)

    # 1b. Create a profile with username for the user (required for store agent lookup)
    username = user.email.split("@")[0]
    await prisma.profile.create(
        data={
            "userId": user.id,
            "username": username,
            "name": f"Test User {username}",
            "description": "Test user profile",
            "links": [],  # Required field - empty array for test profiles
        }
    )

    # 2. Create a test graph with agent input -> agent output
    graph_id = str(uuid.uuid4())

    # Create input node
    input_node_id = str(uuid.uuid4())
    input_block = AgentInputBlock()
    input_node = Node(
        id=input_node_id,
        block_id=input_block.id,
        input_default={
            "name": "test_input",
            "title": "Test Input",
            "value": "",
            "advanced": False,
            "description": "Test input field",
            "placeholder_values": [],
        },
        metadata={"position": {"x": 0, "y": 0}},
    )

    # Create output node
    output_node_id = str(uuid.uuid4())
    output_block = AgentOutputBlock()
    output_node = Node(
        id=output_node_id,
        block_id=output_block.id,
        input_default={
            "name": "test_output",
            "title": "Test Output",
            "value": "",
            "format": "",
            "advanced": False,
            "description": "Test output field",
        },
        metadata={"position": {"x": 200, "y": 0}},
    )

    # Create link from input to output
    link = Link(
        source_id=input_node_id,
        sink_id=output_node_id,
        source_name="result",
        sink_name="value",
        is_static=True,
    )

    # Create the graph
    graph = Graph(
        id=graph_id,
        version=1,
        is_active=True,
        name="Test Agent",
        description="A simple test agent for testing",
        nodes=[input_node, output_node],
        links=[link],
    )

    created_graph = await create_graph(graph, user.id)

    # 3. Create a store listing and store listing version for the agent
    # Use unique slug to avoid constraint violations
    unique_slug = f"test-agent-{str(uuid.uuid4())[:8]}"
    store_submission = await store_db.create_store_submission(
        user_id=user.id,
        agent_id=created_graph.id,
        agent_version=created_graph.version,
        slug=unique_slug,
        name="Test Agent",
        description="A simple test agent",
        sub_heading="Test agent for unit tests",
        categories=["testing"],
        image_urls=["https://example.com/image.jpg"],
    )

    assert store_submission.store_listing_version_id is not None
    # 4. Approve the store listing version
    await store_db.review_store_submission(
        store_listing_version_id=store_submission.store_listing_version_id,
        is_approved=True,
        external_comments="Approved for testing",
        internal_comments="Test approval",
        reviewer_id=user.id,
    )

    return {
        "user": user,
        "graph": created_graph,
        "store_submission": store_submission,
    }


@pytest.fixture(scope="session")
async def setup_llm_test_data():
    """
    Set up test data for LLM agent tests:
    1. Create a test user
    2. Create test OpenAI credentials for the user
    3. Create a test graph with input -> LLM block -> output
    4. Create and approve a store listing
    """
    key = getenv("OPENAI_API_KEY")
    if not key:
        return pytest.skip("OPENAI_API_KEY is not set")

    # 1. Create a test user
    user_data = {
        "sub": f"test-user-{uuid.uuid4()}",
        "email": f"test-{uuid.uuid4()}@example.com",
    }
    user = await get_or_create_user(user_data)

    # 1b. Create a profile with username for the user (required for store agent lookup)
    username = user.email.split("@")[0]
    await prisma.profile.create(
        data={
            "userId": user.id,
            "username": username,
            "name": f"Test User {username}",
            "description": "Test user profile for LLM tests",
            "links": [],  # Required field - empty array for test profiles
        }
    )

    # 2. Create test OpenAI credentials for the user
    credentials = APIKeyCredentials(
        id=str(uuid.uuid4()),
        provider="openai",
        api_key=SecretStr("test-openai-api-key"),
        title="Test OpenAI API Key",
        expires_at=None,
    )

    # Store the credentials
    creds_store = IntegrationCredentialsStore()
    await creds_store.add_creds(user.id, credentials)

    # 3. Create a test graph with input -> LLM block -> output
    graph_id = str(uuid.uuid4())

    # Create input node for the prompt
    input_node_id = str(uuid.uuid4())
    input_block = AgentInputBlock()
    input_node = Node(
        id=input_node_id,
        block_id=input_block.id,
        input_default={
            "name": "user_prompt",
            "title": "User Prompt",
            "value": "",
            "advanced": False,
            "description": "Prompt for the LLM",
            "placeholder_values": [],
        },
        metadata={"position": {"x": 0, "y": 0}},
    )

    # Create LLM block node
    llm_node_id = str(uuid.uuid4())
    llm_block = AITextGeneratorBlock()
    llm_node = Node(
        id=llm_node_id,
        block_id=llm_block.id,
        input_default={
            "model": "gpt-4o-mini",
            "sys_prompt": "You are a helpful assistant.",
            "retry": 3,
            "prompt_values": {},
            "credentials": {
                "provider": "openai",
                "id": credentials.id,
                "type": "api_key",
                "title": credentials.title,
            },
        },
        metadata={"position": {"x": 300, "y": 0}},
    )

    # Create output node
    output_node_id = str(uuid.uuid4())
    output_block = AgentOutputBlock()
    output_node = Node(
        id=output_node_id,
        block_id=output_block.id,
        input_default={
            "name": "llm_response",
            "title": "LLM Response",
            "value": "",
            "format": "",
            "advanced": False,
            "description": "Response from the LLM",
        },
        metadata={"position": {"x": 600, "y": 0}},
    )

    # Create links
    # Link input.result -> llm.prompt
    link1 = Link(
        source_id=input_node_id,
        sink_id=llm_node_id,
        source_name="result",
        sink_name="prompt",
        is_static=True,
    )

    # Link llm.response -> output.value
    link2 = Link(
        source_id=llm_node_id,
        sink_id=output_node_id,
        source_name="response",
        sink_name="value",
        is_static=False,
    )

    # Create the graph
    graph = Graph(
        id=graph_id,
        version=1,
        is_active=True,
        name="LLM Test Agent",
        description="An agent that uses an LLM to process text",
        nodes=[input_node, llm_node, output_node],
        links=[link1, link2],
    )

    created_graph = await create_graph(graph, user.id)

    # 4. Create and approve a store listing
    unique_slug = f"llm-test-agent-{str(uuid.uuid4())[:8]}"
    store_submission = await store_db.create_store_submission(
        user_id=user.id,
        agent_id=created_graph.id,
        agent_version=created_graph.version,
        slug=unique_slug,
        name="LLM Test Agent",
        description="An agent with LLM capabilities",
        sub_heading="Test agent with OpenAI integration",
        categories=["testing", "ai"],
        image_urls=["https://example.com/image.jpg"],
    )
    assert store_submission.store_listing_version_id is not None
    await store_db.review_store_submission(
        store_listing_version_id=store_submission.store_listing_version_id,
        is_approved=True,
        external_comments="Approved for testing",
        internal_comments="Test approval for LLM agent",
        reviewer_id=user.id,
    )

    return {
        "user": user,
        "graph": created_graph,
        "credentials": credentials,
        "store_submission": store_submission,
    }


@pytest.fixture(scope="session")
async def setup_firecrawl_test_data():
    """
    Set up test data for Firecrawl agent tests (missing credentials scenario):
    1. Create a test user (WITHOUT Firecrawl credentials)
    2. Create a test graph with input -> Firecrawl block -> output
    3. Create and approve a store listing
    """
    # 1. Create a test user
    user_data = {
        "sub": f"test-user-{uuid.uuid4()}",
        "email": f"test-{uuid.uuid4()}@example.com",
    }
    user = await get_or_create_user(user_data)

    # 1b. Create a profile with username for the user (required for store agent lookup)
    username = user.email.split("@")[0]
    await prisma.profile.create(
        data={
            "userId": user.id,
            "username": username,
            "name": f"Test User {username}",
            "description": "Test user profile for Firecrawl tests",
            "links": [],  # Required field - empty array for test profiles
        }
    )

    # NOTE: We deliberately do NOT create Firecrawl credentials for this user
    # This tests the scenario where required credentials are missing

    # 2. Create a test graph with input -> Firecrawl block -> output
    graph_id = str(uuid.uuid4())

    # Create input node for the URL
    input_node_id = str(uuid.uuid4())
    input_block = AgentInputBlock()
    input_node = Node(
        id=input_node_id,
        block_id=input_block.id,
        input_default={
            "name": "url",
            "title": "URL to Scrape",
            "value": "",
            "advanced": False,
            "description": "URL for Firecrawl to scrape",
            "placeholder_values": [],
        },
        metadata={"position": {"x": 0, "y": 0}},
    )

    # Create Firecrawl block node
    firecrawl_node_id = str(uuid.uuid4())
    firecrawl_block = FirecrawlScrapeBlock()
    firecrawl_node = Node(
        id=firecrawl_node_id,
        block_id=firecrawl_block.id,
        input_default={
            "limit": 10,
            "only_main_content": True,
            "max_age": 3600000,
            "wait_for": 200,
            "formats": ["markdown"],
            "credentials": {
                "provider": "firecrawl",
                "id": "test-firecrawl-id",
                "type": "api_key",
                "title": "Firecrawl API Key",
            },
        },
        metadata={"position": {"x": 300, "y": 0}},
    )

    # Create output node
    output_node_id = str(uuid.uuid4())
    output_block = AgentOutputBlock()
    output_node = Node(
        id=output_node_id,
        block_id=output_block.id,
        input_default={
            "name": "scraped_data",
            "title": "Scraped Data",
            "value": "",
            "format": "",
            "advanced": False,
            "description": "Data scraped by Firecrawl",
        },
        metadata={"position": {"x": 600, "y": 0}},
    )

    # Create links
    # Link input.result -> firecrawl.url
    link1 = Link(
        source_id=input_node_id,
        sink_id=firecrawl_node_id,
        source_name="result",
        sink_name="url",
        is_static=True,
    )

    # Link firecrawl.markdown -> output.value
    link2 = Link(
        source_id=firecrawl_node_id,
        sink_id=output_node_id,
        source_name="markdown",
        sink_name="value",
        is_static=False,
    )

    # Create the graph
    graph = Graph(
        id=graph_id,
        version=1,
        is_active=True,
        name="Firecrawl Test Agent",
        description="An agent that uses Firecrawl to scrape websites",
        nodes=[input_node, firecrawl_node, output_node],
        links=[link1, link2],
    )

    created_graph = await create_graph(graph, user.id)

    # 3. Create and approve a store listing
    unique_slug = f"firecrawl-test-agent-{str(uuid.uuid4())[:8]}"
    store_submission = await store_db.create_store_submission(
        user_id=user.id,
        agent_id=created_graph.id,
        agent_version=created_graph.version,
        slug=unique_slug,
        name="Firecrawl Test Agent",
        description="An agent with Firecrawl integration (no credentials)",
        sub_heading="Test agent requiring Firecrawl credentials",
        categories=["testing", "scraping"],
        image_urls=["https://example.com/image.jpg"],
    )
    assert store_submission.store_listing_version_id is not None
    await store_db.review_store_submission(
        store_listing_version_id=store_submission.store_listing_version_id,
        is_approved=True,
        external_comments="Approved for testing",
        internal_comments="Test approval for Firecrawl agent",
        reviewer_id=user.id,
    )

    return {
        "user": user,
        "graph": created_graph,
        "store_submission": store_submission,
    }
