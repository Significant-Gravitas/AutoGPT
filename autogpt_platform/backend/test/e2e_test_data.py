"""
E2E Test Data Creator for AutoGPT Platform

This script creates test data for E2E tests by using API functions instead of direct Prisma calls.
This approach ensures compatibility with future model changes by using the API layer.

Image/Video URL Domains Used:
- Images: picsum.photos (for all image URLs - avatars, store listing images, etc.)
- Videos: youtube.com (for store listing video URLs)

Add these domains to your Next.js config:
```javascript
// next.config.js
images: {
  domains: ['picsum.photos'],
}
```
"""

import asyncio
import random
from typing import Any, Dict, List

from faker import Faker

from backend.data.api_key import generate_api_key
from backend.data.credit import get_user_credit_model
from backend.data.db import prisma
from backend.data.graph import Graph, Link, Node, create_graph

# Import API functions from the backend
from backend.data.user import get_or_create_user
from backend.server.integrations.utils import get_supabase
from backend.server.v2.library.db import create_library_agent, create_preset
from backend.server.v2.library.model import LibraryAgentPresetCreatable
from backend.server.v2.store.db import create_store_submission, review_store_submission

faker = Faker()


# Constants for data generation limits (reduced for E2E tests)
NUM_USERS = 10
NUM_AGENT_BLOCKS = 20
MIN_GRAPHS_PER_USER = 10
MAX_GRAPHS_PER_USER = 10
MIN_NODES_PER_GRAPH = 2
MAX_NODES_PER_GRAPH = 4
MIN_PRESETS_PER_USER = 1
MAX_PRESETS_PER_USER = 2
MIN_AGENTS_PER_USER = 10
MAX_AGENTS_PER_USER = 10
MIN_EXECUTIONS_PER_GRAPH = 1
MAX_EXECUTIONS_PER_GRAPH = 5
MIN_REVIEWS_PER_VERSION = 1
MAX_REVIEWS_PER_VERSION = 3


def get_image():
    """Generate a consistent image URL using picsum.photos service."""
    width = random.choice([200, 300, 400, 500, 600, 800])
    height = random.choice([200, 300, 400, 500, 600, 800])
    seed = random.randint(1, 1000)
    return f"https://picsum.photos/seed/{seed}/{width}/{height}"


def get_video_url():
    """Generate a consistent video URL using YouTube."""
    video_ids = [
        "dQw4w9WgXcQ",
        "9bZkp7q19f0",
        "kJQP7kiw5Fk",
        "RgKAFK5djSk",
        "L_jWHffIx5E",
    ]
    video_id = random.choice(video_ids)
    return f"https://www.youtube.com/watch?v={video_id}"


class TestDataCreator:
    """Creates test data using API functions for E2E tests."""

    def __init__(self):
        self.users: List[Dict[str, Any]] = []
        self.agent_blocks: List[Dict[str, Any]] = []
        self.agent_graphs: List[Dict[str, Any]] = []
        self.library_agents: List[Dict[str, Any]] = []
        self.store_submissions: List[Dict[str, Any]] = []
        self.api_keys: List[Dict[str, Any]] = []
        self.presets: List[Dict[str, Any]] = []

    async def create_test_users(self) -> List[Dict[str, Any]]:
        """Create test users using Supabase client."""
        print(f"Creating {NUM_USERS} test users...")

        supabase = get_supabase()
        users = []

        for i in range(NUM_USERS):
            try:
                # Generate test user data
                if i == 0:
                    # First user should have test123@gmail.com email for testing
                    email = "test123@gmail.com"
                else:
                    email = faker.unique.email()
                password = "testpassword123"  # Standard test password
                user_id = f"test-user-{i}-{faker.uuid4()}"

                # Create user in Supabase Auth (if needed)
                try:
                    auth_response = supabase.auth.admin.create_user(
                        {"email": email, "password": password, "email_confirm": True}
                    )
                    if auth_response.user:
                        user_id = auth_response.user.id
                except Exception as supabase_error:
                    print(
                        f"Supabase user creation failed for {email}, using fallback: {supabase_error}"
                    )
                    # Fall back to direct database creation

                # Create mock user data similar to what auth middleware would provide
                user_data = {
                    "sub": user_id,
                    "email": email,
                }

                # Use the API function to create user in local database
                user = await get_or_create_user(user_data)
                users.append(user.model_dump())

            except Exception as e:
                print(f"Error creating user {i}: {e}")
                continue

        self.users = users
        return users

    async def get_available_blocks(self) -> List[Dict[str, Any]]:
        """Get available agent blocks from database."""
        print("Getting available agent blocks...")

        # Get blocks from database instead of the registry
        db_blocks = await prisma.agentblock.find_many()
        if not db_blocks:
            print("No blocks found in database, creating some basic blocks...")
            # Create some basic blocks if none exist
            from backend.blocks.io import AgentInputBlock, AgentOutputBlock
            from backend.blocks.maths import CalculatorBlock
            from backend.blocks.time_blocks import GetCurrentTimeBlock

            blocks_to_create = [
                AgentInputBlock(),
                AgentOutputBlock(),
                CalculatorBlock(),
                GetCurrentTimeBlock(),
            ]

            for block in blocks_to_create:
                try:
                    await prisma.agentblock.create(
                        data={
                            "id": block.id,
                            "name": block.name,
                            "inputSchema": "{}",
                            "outputSchema": "{}",
                        }
                    )
                except Exception as e:
                    print(f"Error creating block {block.name}: {e}")

            # Get blocks again after creation
            db_blocks = await prisma.agentblock.find_many()

        self.agent_blocks = [
            {"id": block.id, "name": block.name} for block in db_blocks
        ]
        print(f"Found {len(self.agent_blocks)} blocks in database")
        return self.agent_blocks

    async def create_test_graphs(self) -> List[Dict[str, Any]]:
        """Create test graphs using the API function."""
        print("Creating test graphs...")

        graphs = []
        for user in self.users:
            num_graphs = random.randint(MIN_GRAPHS_PER_USER, MAX_GRAPHS_PER_USER)

            for graph_num in range(num_graphs):
                # Create a simple graph with nodes and links
                graph_id = str(faker.uuid4())
                nodes = []
                links = []

                # Determine if this should be a DummyInput graph (first 3-4 graphs per user)
                is_dummy_input = graph_num < 4

                # Create nodes based on graph type
                if is_dummy_input:
                    # For dummy input graphs: only GetCurrentTimeBlock
                    node_id = str(faker.uuid4())
                    block = next(
                        b
                        for b in self.agent_blocks
                        if b["name"] == "GetCurrentTimeBlock"
                    )
                    input_default = {"trigger": "start", "format": "%H:%M:%S"}

                    node = Node(
                        id=node_id,
                        block_id=block["id"],
                        input_default=input_default,
                        metadata={"position": {"x": 0, "y": 0}},
                    )
                    nodes.append(node)
                else:
                    # For regular graphs: Create calculator agent pattern with 4 nodes
                    # Node 1: AgentInputBlock for 'a'
                    input_a_id = str(faker.uuid4())
                    input_a_block = next(
                        b for b in self.agent_blocks if b["name"] == "AgentInputBlock"
                    )
                    input_a_node = Node(
                        id=input_a_id,
                        block_id=input_a_block["id"],
                        input_default={
                            "name": "a",
                            "title": None,
                            "value": "",
                            "advanced": False,
                            "description": None,
                            "placeholder_values": [],
                        },
                        metadata={"position": {"x": -1012, "y": 674}},
                    )
                    nodes.append(input_a_node)

                    # Node 2: AgentInputBlock for 'b'
                    input_b_id = str(faker.uuid4())
                    input_b_block = next(
                        b for b in self.agent_blocks if b["name"] == "AgentInputBlock"
                    )
                    input_b_node = Node(
                        id=input_b_id,
                        block_id=input_b_block["id"],
                        input_default={
                            "name": "b",
                            "title": None,
                            "value": "",
                            "advanced": False,
                            "description": None,
                            "placeholder_values": [],
                        },
                        metadata={"position": {"x": -1117, "y": 78}},
                    )
                    nodes.append(input_b_node)

                    # Node 3: CalculatorBlock
                    calc_id = str(faker.uuid4())
                    calc_block = next(
                        b for b in self.agent_blocks if b["name"] == "CalculatorBlock"
                    )
                    calc_node = Node(
                        id=calc_id,
                        block_id=calc_block["id"],
                        input_default={"operation": "Add", "round_result": False},
                        metadata={"position": {"x": -435, "y": 363}},
                    )
                    nodes.append(calc_node)

                    # Node 4: AgentOutputBlock
                    output_id = str(faker.uuid4())
                    output_block = next(
                        b for b in self.agent_blocks if b["name"] == "AgentOutputBlock"
                    )
                    output_node = Node(
                        id=output_id,
                        block_id=output_block["id"],
                        input_default={
                            "name": "result",
                            "title": None,
                            "value": "",
                            "format": "",
                            "advanced": False,
                            "description": None,
                        },
                        metadata={"position": {"x": 402, "y": 0}},
                    )
                    nodes.append(output_node)

                    # Create links between nodes (only for non-dummy graphs with multiple nodes)
                    if len(nodes) >= 4:
                        # Use the actual node IDs from the created nodes instead of our variables
                        actual_input_a_id = nodes[0].id  # First node (input_a)
                        actual_input_b_id = nodes[1].id  # Second node (input_b)
                        actual_calc_id = nodes[2].id  # Third node (calculator)
                        actual_output_id = nodes[3].id  # Fourth node (output)

                        # Link input_a to calculator.a
                        link1 = Link(
                            source_id=actual_input_a_id,
                            sink_id=actual_calc_id,
                            source_name="result",
                            sink_name="a",
                            is_static=True,
                        )
                        links.append(link1)

                        # Link input_b to calculator.b
                        link2 = Link(
                            source_id=actual_input_b_id,
                            sink_id=actual_calc_id,
                            source_name="result",
                            sink_name="b",
                            is_static=True,
                        )
                        links.append(link2)

                        # Link calculator.result to output.value
                        link3 = Link(
                            source_id=actual_calc_id,
                            sink_id=actual_output_id,
                            source_name="result",
                            sink_name="value",
                            is_static=False,
                        )
                        links.append(link3)

                # Create graph object with DummyInput in name if it's a dummy input graph
                graph_name = faker.sentence(nb_words=3)
                if is_dummy_input:
                    graph_name = f"DummyInput {graph_name}"

                graph = Graph(
                    id=graph_id,
                    name=graph_name,
                    description=faker.text(max_nb_chars=200),
                    nodes=nodes,
                    links=links,
                    is_active=True,
                )

                try:
                    # Use the API function to create graph
                    created_graph = await create_graph(graph, user["id"])
                    graph_dict = created_graph.model_dump()
                    # Ensure userId is included for store submissions
                    graph_dict["userId"] = user["id"]
                    graphs.append(graph_dict)
                    print(
                        f"✅ Created graph for user {user['id']}: {graph_dict['name']}"
                    )
                except Exception as e:
                    print(f"Error creating graph: {e}")
                    continue

        self.agent_graphs = graphs
        return graphs

    async def create_test_library_agents(self) -> List[Dict[str, Any]]:
        """Create test library agents using the API function."""
        print("Creating test library agents...")

        library_agents = []
        for user in self.users:
            num_agents = 10  # Create exactly 10 agents per user

            # Get available graphs for this user
            user_graphs = [
                g for g in self.agent_graphs if g.get("userId") == user["id"]
            ]
            if not user_graphs:
                continue

            # Shuffle and take unique graphs to avoid duplicates
            random.shuffle(user_graphs)
            selected_graphs = user_graphs[: min(num_agents, len(user_graphs))]

            for graph_data in selected_graphs:
                try:
                    # Get the graph model from the database
                    from backend.data.graph import get_graph

                    graph = await get_graph(
                        graph_data["id"], graph_data.get("version", 1), user["id"]
                    )
                    if graph:
                        # Use the API function to create library agent
                        library_agents.extend(
                            v.model_dump()
                            for v in await create_library_agent(graph, user["id"])
                        )
                except Exception as e:
                    print(f"Error creating library agent: {e}")
                    continue

        self.library_agents = library_agents
        return library_agents

    async def create_test_presets(self) -> List[Dict[str, Any]]:
        """Create test presets using the API function."""
        print("Creating test presets...")

        presets = []
        for user in self.users:
            num_presets = random.randint(MIN_PRESETS_PER_USER, MAX_PRESETS_PER_USER)

            # Get available graphs for this user
            user_graphs = [
                g for g in self.agent_graphs if g.get("userId") == user["id"]
            ]
            if not user_graphs:
                continue

            for _ in range(min(num_presets, len(user_graphs))):
                graph = random.choice(user_graphs)

                preset_data = LibraryAgentPresetCreatable(
                    name=faker.sentence(nb_words=3),
                    description=faker.text(max_nb_chars=200),
                    graph_id=graph["id"],  # Fixed field name
                    graph_version=graph.get("version", 1),  # Fixed field name
                    inputs={},  # Required field - empty inputs for test data
                    credentials={},  # Required field - empty credentials for test data
                    is_active=True,
                )

                try:
                    # Use the API function to create preset
                    preset = await create_preset(user["id"], preset_data)
                    presets.append(preset.model_dump())
                except Exception as e:
                    print(f"Error creating preset: {e}")
                    continue

        self.presets = presets
        return presets

    async def create_test_api_keys(self) -> List[Dict[str, Any]]:
        """Create test API keys using the API function."""
        print("Creating test API keys...")

        api_keys = []
        for user in self.users:
            from backend.data.api_key import APIKeyPermission

            try:
                # Use the API function to create API key
                api_key, _ = await generate_api_key(
                    name=faker.word(),
                    user_id=user["id"],
                    permissions=[
                        APIKeyPermission.EXECUTE_GRAPH,
                        APIKeyPermission.READ_GRAPH,
                    ],
                    description=faker.text(),
                )
                api_keys.append(api_key.model_dump())
            except Exception as e:
                print(f"Error creating API key for user {user['id']}: {e}")
                continue

        self.api_keys = api_keys
        return api_keys

    async def create_test_store_submissions(self) -> List[Dict[str, Any]]:
        """Create test store submissions using the API function."""
        print("Creating test store submissions...")

        submissions = []
        approved_submissions = []

        for user in self.users:
            # Get available graphs for this specific user
            user_graphs = [
                g for g in self.agent_graphs if g.get("userId") == user["id"]
            ]
            print(f"User {user['id']} has {len(user_graphs)} graphs")
            if not user_graphs:
                print(
                    f"No graphs found for user {user['id']}, skipping store submissions"
                )
                continue

            # Create exactly 4 store submissions per user
            for _ in range(4):
                graph = random.choice(user_graphs)

                try:
                    print(
                        f"Creating store submission for user {user['id']} with graph {graph['id']} (owner: {graph.get('userId')})"
                    )

                    # Use the API function to create store submission with correct parameters
                    submission = await create_store_submission(
                        user_id=user["id"],  # Must match graph's userId
                        agent_id=graph["id"],
                        agent_version=graph.get("version", 1),
                        slug=faker.slug(),
                        name=graph.get("name", faker.sentence(nb_words=3)),
                        sub_heading=faker.sentence(),
                        video_url=get_video_url() if random.random() < 0.3 else None,
                        image_urls=[get_image() for _ in range(3)],
                        description=faker.text(),
                        categories=[faker.word() for _ in range(3)],
                        changes_summary="Initial E2E test submission",
                    )
                    submissions.append(submission.model_dump())
                    print(f"✅ Created store submission: {submission.name}")

                    # Approve the submission so it appears in the store
                    if submission.store_listing_version_id:
                        try:
                            # Pick a random user as the reviewer (admin)
                            reviewer_id = random.choice(self.users)["id"]

                            approved_submission = await review_store_submission(
                                store_listing_version_id=submission.store_listing_version_id,
                                is_approved=True,
                                external_comments="Auto-approved for E2E testing",
                                internal_comments="Automatically approved by E2E test data script",
                                reviewer_id=reviewer_id,
                            )
                            approved_submissions.append(
                                approved_submission.model_dump()
                            )
                            print(f"✅ Approved store submission: {submission.name}")
                        except Exception as e:
                            print(
                                f"Warning: Could not approve submission {submission.name}: {e}"
                            )

                except Exception as e:
                    print(
                        f"Error creating store submission for user {user['id']} graph {graph['id']}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    continue

        print(
            f"Created {len(submissions)} store submissions, approved {len(approved_submissions)}"
        )
        self.store_submissions = submissions
        return submissions

    async def add_user_credits(self):
        """Add credits to users."""
        print("Adding credits to users...")

        credit_model = get_user_credit_model()

        for user in self.users:
            try:
                # Skip credits for disabled credit model to avoid errors
                if (
                    hasattr(credit_model, "__class__")
                    and "Disabled" in credit_model.__class__.__name__
                ):
                    print(f"Skipping credits for user {user['id']} - credits disabled")
                    continue

                # Add random credits to each user
                credit_amount = random.randint(100, 1000)

                await credit_model.top_up_credits(
                    user_id=user["id"], amount=credit_amount
                )
                print(f"Added {credit_amount} credits to user {user['id']}")
            except Exception:
                print(
                    f"Skipping credits for user {user['id']}: credits may be disabled"
                )
                continue

    async def create_all_test_data(self):
        """Create all test data."""
        print("Starting E2E test data creation...")

        # Create users first
        await self.create_test_users()

        # Get available blocks
        await self.get_available_blocks()

        # Create graphs
        await self.create_test_graphs()

        # Create library agents
        await self.create_test_library_agents()

        # Create presets
        await self.create_test_presets()

        # Create API keys
        await self.create_test_api_keys()

        # Create store submissions
        await self.create_test_store_submissions()

        # Add user credits
        await self.add_user_credits()

        # Refresh materialized views
        print("Refreshing materialized views...")
        try:
            await prisma.execute_raw("SELECT refresh_store_materialized_views();")
        except Exception as e:
            print(f"Error refreshing materialized views: {e}")

        print("E2E test data creation completed successfully!")

        # Print summary
        print("\n🎉 E2E Test Data Creation Summary:")
        print(f"✅ Users created: {len(self.users)}")
        print(f"✅ Agent blocks available: {len(self.agent_blocks)}")
        print(f"✅ Agent graphs created: {len(self.agent_graphs)}")
        print(f"✅ Library agents created: {len(self.library_agents)}")
        print(f"✅ Store submissions created: {len(self.store_submissions)}")
        print(f"✅ API keys created: {len(self.api_keys)}")
        print(f"✅ Presets created: {len(self.presets)}")
        print("\n🚀 Your E2E test database is ready to use!")


async def main():
    """Main function to run the test data creation."""
    # Connect to database
    await prisma.connect()

    try:
        creator = TestDataCreator()
        await creator.create_all_test_data()
    finally:
        # Disconnect from database
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
