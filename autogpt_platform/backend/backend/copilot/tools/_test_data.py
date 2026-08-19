import logging
import uuid
from datetime import UTC, datetime
from os import getenv

import pytest
import pytest_asyncio
from prisma.types import ProfileCreateInput
from pydantic import SecretStr

from backend.api.features.library import db as library_db
from backend.api.features.store import db as store_db
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.firecrawl.scrape import FirecrawlScrapeBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.blocks.llm import AITextGeneratorBlock
from backend.copilot.model import ChatMessage, ChatSession
from backend.data import db as db_module
from backend.data.db import prisma
from backend.data.graph import Graph, GraphModel, Link, Node, create_graph
from backend.data.model import APIKeyCredentials
from backend.data.user import get_or_create_user
from backend.integrations.credentials_store import IntegrationCredentialsStore

_logger = logging.getLogger(__name__)


async def _ensure_db_connected() -> None:
    """Ensure the Prisma connection is alive on the current event loop.

    On Python 3.11, the httpx transport inside Prisma can reference a stale
    (closed) event loop when session-scoped async fixtures are evaluated long
    after the initial ``server`` fixture connected Prisma.  A cheap health-check
    followed by a reconnect fixes this without affecting other fixtures.
    """
    try:
        await prisma.query_raw("SELECT 1")
    except Exception:
        _logger.info("Prisma connection stale – reconnecting")
        try:
            await db_module.disconnect()
        except Exception:
            pass
        await db_module.connect()


def make_session(
    user_id: str,
    *,
    guide_read: bool = True,
    library_check: bool = True,
    expert_id: str | None = None,
):
    """Build a fake ChatSession for tool tests.

    ``guide_read=True`` (default) pre-populates the session with a
    ``get_agent_building_guide`` tool-call history entry so the agent-
    generation gate lets through any subsequent ``create_agent`` /
    ``edit_agent`` / ``validate_agent_graph`` / ``fix_agent_graph`` call.

    ``library_check=True`` (default) announces an in-flight
    ``find_library_agent(for_creation=true)`` call so the create-time
    library-similarity gate lets through ``create_agent``. The gate is
    turn-scoped (in-flight only), so seeding via the in-flight buffer —
    not the durable messages list — is the correct shape.
    """
    messages: list[ChatMessage] = []
    if guide_read:
        messages.append(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"function": {"name": "get_agent_building_guide"}}],
            )
        )
    session = ChatSession(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        messages=messages,
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        successful_agent_runs={},
        successful_agent_schedules={},
        expert_id=expert_id,
    )
    if library_check:
        session.announce_inflight_tool_call(
            "find_library_agent",
            arguments={"for_creation": True, "goal_summary": "test"},
        )
    return session


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def setup_test_data(server):
    """
    Set up test data for run_agent tests:
    1. Create a test user
    2. Create a test graph (agent input -> agent output)
    3. Create a store listing and store listing version
    4. Approve the store listing version

    Depends on ``server`` to ensure Prisma is connected.
    """
    await _ensure_db_connected()

    # 1. Create a test user
    user = await _create_user_with_profile("Test user profile")

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

    # 3. Create and approve a store listing
    store_submission = await _publish_to_store(
        user.id,
        created_graph,
        slug_prefix="test-agent",
        name="Test Agent",
        description="A simple test agent",
        sub_heading="Test agent for unit tests",
        categories=["testing"],
    )

    return {
        "user": user,
        "graph": created_graph,
        "store_submission": store_submission,
    }


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def setup_llm_test_data(server):
    """
    Set up test data for LLM agent tests:
    1. Create a test user
    2. Create test OpenAI credentials for the user
    3. Create a test graph with input -> LLM block -> output
    4. Create and approve a store listing

    Depends on ``server`` to ensure Prisma is connected.
    """
    await _ensure_db_connected()

    key = getenv("OPENAI_API_KEY")
    if not key:
        return pytest.skip("OPENAI_API_KEY is not set")

    # 1. Create a test user
    user = await _create_user_with_profile("Test user profile for LLM tests")

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
    store_submission = await _publish_to_store(
        user.id,
        created_graph,
        slug_prefix="llm-test-agent",
        name="LLM Test Agent",
        description="An agent with LLM capabilities",
        sub_heading="Test agent with OpenAI integration",
        categories=["testing", "ai"],
    )

    return {
        "user": user,
        "graph": created_graph,
        "credentials": credentials,
        "store_submission": store_submission,
    }


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def setup_firecrawl_test_data(server):
    """
    Set up test data for Firecrawl agent tests (missing credentials scenario):
    1. Create a test user (WITHOUT Firecrawl credentials)
    2. Create a test graph with input -> Firecrawl block -> output
    3. Create and approve a store listing

    Depends on ``server`` to ensure Prisma is connected.
    """
    await _ensure_db_connected()

    # 1. Create a test user
    user = await _create_user_with_profile("Test user profile for Firecrawl tests")

    # NOTE: We deliberately do NOT create Firecrawl credentials for this user
    # This tests the scenario where required credentials are missing

    # 2. Create a test graph with input -> Firecrawl block -> output
    created_graph = await create_graph(
        _build_firecrawl_graph(
            name="Firecrawl Test Agent",
            description="An agent that uses Firecrawl to scrape websites",
        ),
        user.id,
    )

    # 3. Create and approve a store listing
    store_submission = await _publish_to_store(
        user.id,
        created_graph,
        slug_prefix="firecrawl-test-agent",
        name="Firecrawl Test Agent",
        description="An agent with Firecrawl integration (no credentials)",
        sub_heading="Test agent requiring Firecrawl credentials",
        categories=["testing", "scraping"],
    )

    return {
        "user": user,
        "graph": created_graph,
        "store_submission": store_submission,
    }


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def setup_subagent_test_data(server):
    """
    Orchestrator agent (input -> AgentExecutorBlock -> output) wrapping a
    Firecrawl sub-graph, for a user without Firecrawl credentials. The parent
    has NO credential fields of its own; all of them live in the sub-graph.

    Registered both in the library and the store, to cover both run paths.
    Depends on ``server`` to ensure Prisma is connected.
    """
    await _ensure_db_connected()

    # 1. Create a test user (deliberately without Firecrawl credentials)
    user = await _create_user_with_profile("Test user profile for sub-agent tests")

    # 2. Create the Firecrawl sub-graph
    sub_graph = await create_graph(
        _build_firecrawl_graph(
            name="Firecrawl Sub-Agent",
            description="Sub-agent that scrapes a website with Firecrawl",
        ),
        user.id,
    )

    # 3. Create the parent graph: input -> sub-agent -> output
    input_node_id = str(uuid.uuid4())
    input_node = Node(
        id=input_node_id,
        block_id=AgentInputBlock().id,
        input_default={
            "name": "url",
            "title": "URL to Scrape",
            "value": "",
            "advanced": False,
            "description": "URL to hand to the sub-agent",
        },
        metadata={"position": {"x": 0, "y": 0}},
    )

    sub_agent_node_id = str(uuid.uuid4())
    sub_agent_node = Node(
        id=sub_agent_node_id,
        block_id=AgentExecutorBlock().id,
        input_default={
            # Placeholders filled at execution time, as the builder persists them
            "user_id": "",
            "inputs": {},
            "graph_id": sub_graph.id,
            "graph_version": sub_graph.version,
            "input_schema": sub_graph.input_schema,
            "output_schema": sub_graph.output_schema,
        },
        metadata={"position": {"x": 300, "y": 0}},
    )

    output_node_id = str(uuid.uuid4())
    output_node = Node(
        id=output_node_id,
        block_id=AgentOutputBlock().id,
        input_default={
            "name": "scraped_data",
            "title": "Scraped Data",
            "value": "",
            "format": "",
            "advanced": False,
            "description": "Data scraped by the sub-agent",
        },
        metadata={"position": {"x": 600, "y": 0}},
    )

    parent_graph = await create_graph(
        Graph(
            id=str(uuid.uuid4()),
            version=1,
            is_active=True,
            name="Sub-Agent Orchestrator",
            description="An agent whose only credentials live in its sub-agent",
            nodes=[input_node, sub_agent_node, output_node],
            links=[
                Link(
                    source_id=input_node_id,
                    sink_id=sub_agent_node_id,
                    source_name="result",
                    sink_name="url",
                    is_static=True,
                ),
                Link(
                    source_id=sub_agent_node_id,
                    sink_id=output_node_id,
                    source_name="scraped_data",
                    sink_name="value",
                    is_static=False,
                ),
            ],
        ),
        user.id,
    )

    # 4a. Add the parent to the user's library (the library_agent_id run path)
    library_agents = await library_db.create_library_agent(
        graph=parent_graph,
        user_id=user.id,
        create_library_agents_for_sub_graphs=False,
    )
    assert len(library_agents) == 1

    # 4b. Create and approve a store listing (the marketplace slug run path)
    store_submission = await _publish_to_store(
        user.id,
        parent_graph,
        slug_prefix="subagent-test-agent",
        name="Sub-Agent Orchestrator",
        description="An agent whose only credentials live in its sub-agent",
        sub_heading="Test agent requiring sub-agent credentials",
        categories=["testing"],
    )

    return {
        "user": user,
        "graph": parent_graph,
        "sub_graph": sub_graph,
        "library_agent": library_agents[0],
        "store_submission": store_submission,
    }


async def _create_user_with_profile(profile_description: str):
    """Create a test user + profile. The username is required for store lookups."""
    user = await get_or_create_user(
        {
            "sub": f"test-user-{uuid.uuid4()}",
            "email": f"test-{uuid.uuid4()}@example.com",
        }
    )
    username = user.email.split("@")[0]
    await prisma.profile.upsert(
        where={"userId": user.id},
        data={
            # get_or_create_user auto-creates a default profile; tests need
            # this specific username for store agent lookups.
            "create": ProfileCreateInput(
                userId=user.id,
                username=username,
                name=f"Test User {username}",
                description=profile_description,
                links=[],
            ),
            "update": {
                "username": username,
                "name": f"Test User {username}",
                "description": profile_description,
            },
        },
    )
    return user


async def _publish_to_store(
    user_id: str,
    graph: GraphModel,
    *,
    slug_prefix: str,
    name: str,
    description: str,
    sub_heading: str,
    categories: list[str],
):
    """Submit `graph` to the store and approve it. The slug gets a random
    suffix to avoid constraint violations across runs."""
    submission = await store_db.create_store_submission(
        user_id=user_id,
        graph_id=graph.id,
        graph_version=graph.version,
        slug=f"{slug_prefix}-{str(uuid.uuid4())[:8]}",
        name=name,
        description=description,
        sub_heading=sub_heading,
        categories=categories,
        image_urls=["https://example.com/image.jpg"],
    )
    assert submission.listing_version_id is not None
    await store_db.review_store_submission(
        store_listing_version_id=submission.listing_version_id,
        is_approved=True,
        external_comments="Approved for testing",
        internal_comments=f"Test approval for {name}",
        reviewer_id=user_id,
    )
    return submission


def _build_firecrawl_graph(name: str, description: str) -> Graph:
    """input -> FirecrawlScrapeBlock -> output; needs a Firecrawl API key."""
    input_node_id = str(uuid.uuid4())
    input_node = Node(
        id=input_node_id,
        block_id=AgentInputBlock().id,
        input_default={
            "name": "url",
            "title": "URL to Scrape",
            "value": "",
            "advanced": False,
            "description": "URL for Firecrawl to scrape",
        },
        metadata={"position": {"x": 0, "y": 0}},
    )

    firecrawl_node_id = str(uuid.uuid4())
    firecrawl_node = Node(
        id=firecrawl_node_id,
        block_id=FirecrawlScrapeBlock().id,
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

    output_node_id = str(uuid.uuid4())
    output_node = Node(
        id=output_node_id,
        block_id=AgentOutputBlock().id,
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

    return Graph(
        id=str(uuid.uuid4()),
        version=1,
        is_active=True,
        name=name,
        description=description,
        nodes=[input_node, firecrawl_node, output_node],
        links=[
            Link(
                source_id=input_node_id,
                sink_id=firecrawl_node_id,
                source_name="result",
                sink_name="url",
                is_static=True,
            ),
            Link(
                source_id=firecrawl_node_id,
                sink_id=output_node_id,
                source_name="markdown",
                sink_name="value",
                is_static=False,
            ),
        ],
    )
