"""
Test Data Creator for AutoGPT Platform

This script creates test data for the AutoGPT platform database.

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
from datetime import datetime

import prisma.enums
from autogpt_libs.api_key.keysmith import APIKeySmith
from faker import Faker
from prisma import Json, Prisma
from prisma.types import (
    AgentBlockCreateInput,
    AgentGraphCreateInput,
    AgentNodeCreateInput,
    AgentNodeLinkCreateInput,
    AnalyticsDetailsCreateInput,
    AnalyticsMetricsCreateInput,
    CreditTransactionCreateInput,
    IntegrationWebhookCreateInput,
    ProfileCreateInput,
    StoreListingReviewCreateInput,
    UserCreateInput,
)

faker = Faker()

# Constants for data generation limits

# Base entities
NUM_USERS = 100  # Creates 100 user records
NUM_AGENT_BLOCKS = 100  # Creates 100 agent block templates

# Per-user entities
MIN_GRAPHS_PER_USER = 1  # Each user will have between 1-5 graphs
MAX_GRAPHS_PER_USER = 5  # Total graphs: 500-2500 (NUM_USERS * MIN/MAX_GRAPHS)

# Per-graph entities
MIN_NODES_PER_GRAPH = 2  # Each graph will have between 2-5 nodes
MAX_NODES_PER_GRAPH = (
    5  # Total nodes: 1000-2500 (GRAPHS_PER_USER * NUM_USERS * MIN/MAX_NODES)
)

# Additional per-user entities
MIN_PRESETS_PER_USER = 1  # Each user will have between 1-2 presets
MAX_PRESETS_PER_USER = 5  # Total presets: 500-2500 (NUM_USERS * MIN/MAX_PRESETS)
MIN_AGENTS_PER_USER = 1  # Each user will have between 1-2 agents
MAX_AGENTS_PER_USER = 10  # Total agents: 500-5000 (NUM_USERS * MIN/MAX_AGENTS)

# Execution and review records
MIN_EXECUTIONS_PER_GRAPH = 1  # Each graph will have between 1-5 execution records
MAX_EXECUTIONS_PER_GRAPH = (
    20  # Total executions: 1000-5000 (TOTAL_GRAPHS * MIN/MAX_EXECUTIONS)
)
MIN_REVIEWS_PER_VERSION = 1  # Each version will have between 1-3 reviews
MAX_REVIEWS_PER_VERSION = 5  # Total reviews depends on number of versions created


def get_image():
    """Generate a consistent image URL using picsum.photos service."""
    width = random.choice([200, 300, 400, 500, 600, 800])
    height = random.choice([200, 300, 400, 500, 600, 800])
    # Use a random seed to get different images
    seed = random.randint(1, 1000)
    return f"https://picsum.photos/seed/{seed}/{width}/{height}"


def get_video_url():
    """Generate a consistent video URL using a placeholder service."""
    # Using YouTube as a consistent source for video URLs
    video_ids = [
        "dQw4w9WgXcQ",  # Example video IDs
        "9bZkp7q19f0",
        "kJQP7kiw5Fk",
        "RgKAFK5djSk",
        "L_jWHffIx5E",
    ]
    video_id = random.choice(video_ids)
    return f"https://www.youtube.com/watch?v={video_id}"


async def main():
    db = Prisma()
    await db.connect()

    # Insert Users
    print(f"Inserting {NUM_USERS} users")
    users = []
    for _ in range(NUM_USERS):
        user = await db.user.create(
            data=UserCreateInput(
                id=str(faker.uuid4()),
                email=faker.unique.email(),
                name=faker.name(),
                metadata=prisma.Json({}),
                integrations="",
            )
        )
        users.append(user)

    # Insert AgentBlocks
    agent_blocks = []
    print(f"Inserting {NUM_AGENT_BLOCKS} agent blocks")
    for _ in range(NUM_AGENT_BLOCKS):
        block = await db.agentblock.create(
            data=AgentBlockCreateInput(
                name=f"{faker.word()}_{str(faker.uuid4())[:8]}",
                inputSchema="{}",
                outputSchema="{}",
            )
        )
        agent_blocks.append(block)

    # Insert AgentGraphs
    agent_graphs = []
    print(f"Inserting {NUM_USERS * MAX_GRAPHS_PER_USER} agent graphs")
    for user in users:
        for _ in range(
            random.randint(MIN_GRAPHS_PER_USER, MAX_GRAPHS_PER_USER)
        ):  # Adjust the range to create more graphs per user if desired
            graph = await db.agentgraph.create(
                data=AgentGraphCreateInput(
                    name=faker.sentence(nb_words=3),
                    description=faker.text(max_nb_chars=200),
                    userId=user.id,
                    isActive=True,
                )
            )
            agent_graphs.append(graph)

    # Insert AgentNodes
    agent_nodes = []
    print(
        f"Inserting {NUM_USERS * MAX_GRAPHS_PER_USER * MAX_NODES_PER_GRAPH} agent nodes"
    )
    for graph in agent_graphs:
        num_nodes = random.randint(MIN_NODES_PER_GRAPH, MAX_NODES_PER_GRAPH)
        for _ in range(num_nodes):  # Create 5 AgentNodes per graph
            block = random.choice(agent_blocks)
            node = await db.agentnode.create(
                data=AgentNodeCreateInput(
                    agentBlockId=block.id,
                    agentGraphId=graph.id,
                    agentGraphVersion=graph.version,
                    constantInput=Json({}),
                    metadata=Json({}),
                )
            )
            agent_nodes.append(node)

    # Insert AgentPresets
    agent_presets = []
    print(f"Inserting {NUM_USERS * MAX_PRESETS_PER_USER} agent presets")
    for user in users:
        num_presets = random.randint(MIN_PRESETS_PER_USER, MAX_PRESETS_PER_USER)
        for _ in range(num_presets):  # Create 1 AgentPreset per user
            graph = random.choice(agent_graphs)
            preset = await db.agentpreset.create(
                data={
                    "name": faker.sentence(nb_words=3),
                    "description": faker.text(max_nb_chars=200),
                    "userId": user.id,
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "isActive": True,
                }
            )
            agent_presets.append(preset)

    # Insert Profiles first (before LibraryAgents)
    profiles = []
    print(f"Inserting {NUM_USERS} profiles")
    for user in users:
        profile = await db.profile.create(
            data=ProfileCreateInput(
                userId=user.id,
                name=user.name or faker.name(),
                username=faker.unique.user_name(),
                description=faker.text(),
                links=[faker.url() for _ in range(3)],
                avatarUrl=get_image(),
            )
        )
        profiles.append(profile)

    # Insert LibraryAgents
    library_agents = []
    print("Inserting library agents")
    for user in users:
        num_agents = random.randint(MIN_AGENTS_PER_USER, MAX_AGENTS_PER_USER)
        # Get a shuffled list of graphs to ensure uniqueness per user
        available_graphs = agent_graphs.copy()
        random.shuffle(available_graphs)

        # Limit to available unique graphs
        num_agents = min(num_agents, len(available_graphs))

        for i in range(num_agents):
            graph = available_graphs[i]  # Use unique graph for each library agent

            # Get creator profile for this graph's owner
            creator_profile = next(
                (p for p in profiles if p.userId == graph.userId), None
            )

            library_agent = await db.libraryagent.create(
                data={
                    "userId": user.id,
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "creatorId": creator_profile.id if creator_profile else None,
                    "imageUrl": get_image() if random.random() < 0.5 else None,
                    "useGraphIsActiveVersion": random.choice([True, False]),
                    "isFavorite": random.choice([True, False]),
                    "isCreatedByUser": random.choice([True, False]),
                    "isArchived": random.choice([True, False]),
                    "isDeleted": random.choice([True, False]),
                }
            )
            library_agents.append(library_agent)

    # Insert AgentGraphExecutions
    agent_graph_executions = []
    print(
        f"Inserting {NUM_USERS * MAX_GRAPHS_PER_USER * MAX_EXECUTIONS_PER_GRAPH} agent graph executions"
    )
    graph_execution_data = []
    for graph in agent_graphs:
        user = random.choice(users)
        num_executions = random.randint(
            MIN_EXECUTIONS_PER_GRAPH, MAX_EXECUTIONS_PER_GRAPH
        )
        for _ in range(num_executions):
            matching_presets = [p for p in agent_presets if p.agentGraphId == graph.id]
            preset = (
                random.choice(matching_presets)
                if matching_presets and random.random() < 0.5
                else None
            )

            graph_execution_data.append(
                {
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "userId": user.id,
                    "executionStatus": prisma.enums.AgentExecutionStatus.COMPLETED,
                    "startedAt": faker.date_time_this_year(),
                    "agentPresetId": preset.id if preset else None,
                }
            )

    agent_graph_executions = await db.agentgraphexecution.create_many(
        data=graph_execution_data
    )
    # Need to fetch the created records since create_many doesn't return them
    agent_graph_executions = await db.agentgraphexecution.find_many()

    # Insert AgentNodeExecutions
    print(
        f"Inserting {NUM_USERS * MAX_GRAPHS_PER_USER * MAX_EXECUTIONS_PER_GRAPH} agent node executions"
    )
    node_execution_data = []
    for execution in agent_graph_executions:
        nodes = [
            node for node in agent_nodes if node.agentGraphId == execution.agentGraphId
        ]
        for node in nodes:
            node_execution_data.append(
                {
                    "agentGraphExecutionId": execution.id,
                    "agentNodeId": node.id,
                    "executionStatus": prisma.enums.AgentExecutionStatus.COMPLETED,
                    "addedTime": datetime.now(),
                }
            )

    agent_node_executions = await db.agentnodeexecution.create_many(
        data=node_execution_data
    )
    # Need to fetch the created records since create_many doesn't return them
    agent_node_executions = await db.agentnodeexecution.find_many()

    # Insert AgentNodeExecutionInputOutput
    print(
        f"Inserting {NUM_USERS * MAX_GRAPHS_PER_USER * MAX_EXECUTIONS_PER_GRAPH} agent node execution input/outputs"
    )
    input_output_data = []
    for node_execution in agent_node_executions:
        # Input data
        input_output_data.append(
            {
                "name": "input1",
                "data": "{}",
                "time": datetime.now(),
                "referencedByInputExecId": node_execution.id,
            }
        )
        # Output data
        input_output_data.append(
            {
                "name": "output1",
                "data": "{}",
                "time": datetime.now(),
                "referencedByOutputExecId": node_execution.id,
            }
        )

    await db.agentnodeexecutioninputoutput.create_many(data=input_output_data)

    # Insert AgentNodeLinks
    print(f"Inserting {NUM_USERS * MAX_GRAPHS_PER_USER} agent node links")
    for graph in agent_graphs:
        nodes = [node for node in agent_nodes if node.agentGraphId == graph.id]
        if len(nodes) >= 2:
            source_node = nodes[0]
            sink_node = nodes[1]
            await db.agentnodelink.create(
                data=AgentNodeLinkCreateInput(
                    agentNodeSourceId=source_node.id,
                    sourceName="output1",
                    agentNodeSinkId=sink_node.id,
                    sinkName="input1",
                    isStatic=False,
                )
            )

    # Insert AnalyticsDetails
    print(f"Inserting {NUM_USERS} analytics details")
    for user in users:
        for _ in range(1):
            await db.analyticsdetails.create(
                data=AnalyticsDetailsCreateInput(
                    userId=user.id,
                    type=faker.word(),
                    data=prisma.Json({}),
                    dataIndex=faker.word(),
                )
            )

    # Insert AnalyticsMetrics
    print(f"Inserting {NUM_USERS} analytics metrics")
    for user in users:
        for _ in range(1):
            await db.analyticsmetrics.create(
                data=AnalyticsMetricsCreateInput(
                    userId=user.id,
                    analyticMetric=faker.word(),
                    value=random.uniform(0, 100),
                    dataString=faker.word(),
                )
            )

    # Insert CreditTransaction (formerly UserBlockCredit)
    print(f"Inserting {NUM_USERS} credit transactions")
    for user in users:
        for _ in range(1):
            block = random.choice(agent_blocks)
            await db.credittransaction.create(
                data=CreditTransactionCreateInput(
                    transactionKey=str(faker.uuid4()),
                    userId=user.id,
                    amount=random.randint(1, 100),
                    type=(
                        prisma.enums.CreditTransactionType.TOP_UP
                        if random.random() < 0.5
                        else prisma.enums.CreditTransactionType.USAGE
                    ),
                    metadata=prisma.Json({}),
                )
            )

    # Insert StoreListings
    store_listings = []
    print("Inserting store listings")
    for graph in agent_graphs:
        user = random.choice(users)
        slug = faker.slug()
        listing = await db.storelisting.create(
            data={
                "agentGraphId": graph.id,
                "agentGraphVersion": graph.version,
                "owningUserId": user.id,
                "hasApprovedVersion": random.choice([True, False]),
                "slug": slug,
            }
        )
        store_listings.append(listing)

    # Insert StoreListingVersions
    store_listing_versions = []
    print("Inserting store listing versions")
    for listing in store_listings:
        graph = [g for g in agent_graphs if g.id == listing.agentGraphId][0]
        version = await db.storelistingversion.create(
            data={
                "agentGraphId": graph.id,
                "agentGraphVersion": graph.version,
                "name": graph.name or faker.sentence(nb_words=3),
                "subHeading": faker.sentence(),
                "videoUrl": get_video_url() if random.random() < 0.3 else None,
                "imageUrls": [get_image() for _ in range(3)],
                "description": faker.text(),
                "categories": [faker.word() for _ in range(3)],
                "isFeatured": random.choice([True, False]),
                "isAvailable": True,
                "storeListingId": listing.id,
                "submissionStatus": random.choice(
                    [
                        prisma.enums.SubmissionStatus.PENDING,
                        prisma.enums.SubmissionStatus.APPROVED,
                        prisma.enums.SubmissionStatus.REJECTED,
                    ]
                ),
            }
        )
        store_listing_versions.append(version)

    # Insert StoreListingReviews
    print("Inserting store listing reviews")
    for version in store_listing_versions:
        # Create a copy of users list and shuffle it to avoid duplicates
        available_reviewers = users.copy()
        random.shuffle(available_reviewers)

        # Limit number of reviews to available unique reviewers
        num_reviews = min(
            random.randint(MIN_REVIEWS_PER_VERSION, MAX_REVIEWS_PER_VERSION),
            len(available_reviewers),
        )

        # Take only the first num_reviews reviewers
        for reviewer in available_reviewers[:num_reviews]:
            await db.storelistingreview.create(
                data=StoreListingReviewCreateInput(
                    storeListingVersionId=version.id,
                    reviewByUserId=reviewer.id,
                    score=random.randint(1, 5),
                    comments=faker.text(),
                )
            )

    # Insert UserOnboarding for some users
    print("Inserting user onboarding data")
    for user in random.sample(
        users, k=int(NUM_USERS * 0.7)
    ):  # 70% of users have onboarding data
        completed_steps = []
        possible_steps = list(prisma.enums.OnboardingStep)
        # Randomly complete some steps
        if random.random() < 0.8:
            num_steps = random.randint(1, len(possible_steps))
            completed_steps = random.sample(possible_steps, k=num_steps)

        try:
            await db.useronboarding.create(
                data={
                    "userId": user.id,
                    "completedSteps": completed_steps,
                    "walletShown": random.choice([True, False]),
                    "notified": (
                        random.sample(completed_steps, k=min(3, len(completed_steps)))
                        if completed_steps
                        else []
                    ),
                    "rewardedFor": (
                        random.sample(completed_steps, k=min(2, len(completed_steps)))
                        if completed_steps
                        else []
                    ),
                    "usageReason": (
                        random.choice(["personal", "business", "research", "learning"])
                        if random.random() < 0.7
                        else None
                    ),
                    "integrations": random.sample(
                        ["github", "google", "discord", "slack"], k=random.randint(0, 2)
                    ),
                    "otherIntegrations": (
                        faker.word() if random.random() < 0.2 else None
                    ),
                    "selectedStoreListingVersionId": (
                        random.choice(store_listing_versions).id
                        if store_listing_versions and random.random() < 0.5
                        else None
                    ),
                    "agentInput": (
                        Json({"test": "data"}) if random.random() < 0.3 else None
                    ),
                    "onboardingAgentExecutionId": (
                        random.choice(agent_graph_executions).id
                        if agent_graph_executions and random.random() < 0.3
                        else None
                    ),
                    "agentRuns": random.randint(0, 10),
                }
            )
        except Exception as e:
            print(f"Error creating onboarding for user {user.id}: {e}")
            # Try simpler version
            await db.useronboarding.create(
                data={
                    "userId": user.id,
                }
            )

    # Insert IntegrationWebhooks for some users
    print("Inserting integration webhooks")
    for user in random.sample(
        users, k=int(NUM_USERS * 0.3)
    ):  # 30% of users have webhooks
        for _ in range(random.randint(1, 3)):
            await db.integrationwebhook.create(
                data=IntegrationWebhookCreateInput(
                    userId=user.id,
                    provider=random.choice(["github", "slack", "discord"]),
                    credentialsId=str(faker.uuid4()),
                    webhookType=random.choice(["repo", "channel", "server"]),
                    resource=faker.slug(),
                    events=[
                        random.choice(["created", "updated", "deleted"])
                        for _ in range(random.randint(1, 3))
                    ],
                    config=prisma.Json({"url": faker.url()}),
                    secret=str(faker.sha256()),
                    providerWebhookId=str(faker.uuid4()),
                )
            )

    # Insert APIKeys
    print(f"Inserting {NUM_USERS} api keys")
    for user in users:
        api_key = APIKeySmith().generate_key()
        await db.apikey.create(
            data={
                "name": faker.word(),
                "head": api_key.head,
                "tail": api_key.tail,
                "hash": api_key.hash,
                "salt": api_key.salt,
                "status": prisma.enums.APIKeyStatus.ACTIVE,
                "permissions": [
                    prisma.enums.APIKeyPermission.EXECUTE_GRAPH,
                    prisma.enums.APIKeyPermission.READ_GRAPH,
                ],
                "description": faker.text(),
                "userId": user.id,
            }
        )

    # Refresh materialized views
    print("Refreshing materialized views...")
    await db.execute_raw("SELECT refresh_store_materialized_views();")

    await db.disconnect()
    print("Test data creation completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
