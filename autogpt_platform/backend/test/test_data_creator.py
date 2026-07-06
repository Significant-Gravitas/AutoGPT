"""
Test Data Creator for AutoGPT Platform

This script creates test data for the AutoGPT platform database.

Image/Video URL Domains Used:
- Images: none. Avatars and store listing images are seeded empty so the
  frontend renders its built-in solid-color/boring-avatars fallback, avoiding
  any external image dependency (e.g. picsum.photos).
- Videos: youtube.com (for store listing video URLs)
"""

import asyncio
import os
import random
from datetime import datetime

import prisma.enums
import prisma.models
import pytest
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

# Organizations / teams (tenancy)
NUM_SHARED_ORGS = 5  # Non-personal orgs shared across a subset of users
MIN_SHARED_ORG_MEMBERS = 2  # Additional members beyond the owner
MAX_SHARED_ORG_MEMBERS = 5
MIN_EXTRA_TEAMS_PER_SHARED_ORG = 1  # Extra teams beyond the default team
MAX_EXTRA_TEAMS_PER_SHARED_ORG = 2
MIN_SHARED_ORG_BALANCE = 1000  # Random positive OrgBalance for shared orgs
MAX_SHARED_ORG_BALANCE = 100_000
# Fraction of tenancy-scoped resources routed to a shared org/team the user
# belongs to (instead of the user's personal org) so shared-visibility data exists.
SHARED_TENANCY_RATIO = 0.2


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


async def create_personal_org(db: Prisma, user: prisma.models.User) -> tuple[str, str]:
    """Create a user's personal org + default team.

    Mirrors ``backend.api.features.orgs.db._create_personal_org_for_user``:
    Organization (isPersonal) + owner OrgMember + default Team + TeamMember +
    OrganizationProfile + FREE OrganizationSeatAssignment + zero OrgBalance.
    Returns ``(organization_id, default_team_id)``.
    """
    slug = faker.unique.slug()
    local_part = user.email.split("@")[0] if user.email else "user"
    display_name = user.name or local_part

    org = await db.organization.create(
        data={
            "name": display_name,
            "slug": slug,
            "isPersonal": True,
            "bootstrapUserId": user.id,
            "settings": "{}",
        }
    )
    await db.orgmember.create(
        data={
            "orgId": org.id,
            "userId": user.id,
            "isOwner": True,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )
    team = await db.team.create(
        data={
            "name": "Default",
            "orgId": org.id,
            "isDefault": True,
            "joinPolicy": "OPEN",
            "createdByUserId": user.id,
        }
    )
    await db.teammember.create(
        data={
            "teamId": team.id,
            "userId": user.id,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )
    await db.organizationprofile.create(
        data={
            "organizationId": org.id,
            "username": slug,
            "displayName": display_name,
        }
    )
    await db.organizationseatassignment.create(
        data={
            "organizationId": org.id,
            "userId": user.id,
            "seatType": "FREE",
            "status": "ACTIVE",
            "assignedByUserId": user.id,
        }
    )
    await db.orgbalance.create(data={"orgId": org.id, "balance": 0})
    return org.id, team.id


async def create_shared_org(
    db: Prisma,
    owner: prisma.models.User,
    members: list[prisma.models.User],
) -> tuple[str, list[str], dict[str, list[str]]]:
    """Create a non-personal org shared by ``owner`` + ``members``.

    Adds a default team (everyone) plus 1-2 extra teams (random subsets) and a
    random positive OrgBalance. Returns ``(organization_id, member_ids,
    team_membership)`` where ``team_membership`` maps each team id to the user
    ids that belong to it.
    """
    slug = faker.unique.slug()
    name = faker.company()
    all_members = [owner, *members]

    org = await db.organization.create(
        data={
            "name": name,
            "slug": slug,
            "description": faker.catch_phrase(),
            "isPersonal": False,
            "bootstrapUserId": owner.id,
            "settings": "{}",
        }
    )

    # Owner is owner+admin; the rest are members (a minority promoted to admin).
    await db.orgmember.create(
        data={
            "orgId": org.id,
            "userId": owner.id,
            "isOwner": True,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )
    for member in members:
        await db.orgmember.create(
            data={
                "orgId": org.id,
                "userId": member.id,
                "isAdmin": random.random() < 0.3,
                "status": "ACTIVE",
            }
        )
    for member in all_members:
        await db.organizationseatassignment.create(
            data={
                "organizationId": org.id,
                "userId": member.id,
                "seatType": "FREE",
                "status": "ACTIVE",
                "assignedByUserId": owner.id,
            }
        )

    await db.organizationprofile.create(
        data={
            "organizationId": org.id,
            "username": slug,
            "displayName": name,
        }
    )
    await db.orgbalance.create(
        data={
            "orgId": org.id,
            "balance": random.randint(MIN_SHARED_ORG_BALANCE, MAX_SHARED_ORG_BALANCE),
        }
    )

    team_membership: dict[str, list[str]] = {}

    # Default team contains everyone.
    default_team = await db.team.create(
        data={
            "name": "Default",
            "orgId": org.id,
            "isDefault": True,
            "joinPolicy": "OPEN",
            "createdByUserId": owner.id,
        }
    )
    for member in all_members:
        await db.teammember.create(
            data={
                "teamId": default_team.id,
                "userId": member.id,
                "isAdmin": member.id == owner.id,
                "status": "ACTIVE",
            }
        )
    team_membership[default_team.id] = [m.id for m in all_members]

    # Extra teams get a random subset of members.
    num_extra = random.randint(
        MIN_EXTRA_TEAMS_PER_SHARED_ORG, MAX_EXTRA_TEAMS_PER_SHARED_ORG
    )
    for i in range(num_extra):
        team = await db.team.create(
            data={
                "name": f"{faker.word().capitalize()} Team {i + 1}",
                "orgId": org.id,
                "joinPolicy": random.choice(["OPEN", "PRIVATE"]),
                "createdByUserId": owner.id,
            }
        )
        subset = random.sample(all_members, k=random.randint(1, len(all_members)))
        for member in subset:
            await db.teammember.create(
                data={
                    "teamId": team.id,
                    "userId": member.id,
                    "isAdmin": member.id == owner.id,
                    "status": "ACTIVE",
                }
            )
        team_membership[team.id] = [m.id for m in subset]

    return org.id, [m.id for m in all_members], team_membership


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

    # Insert personal Organizations (one per user) + a handful of shared orgs.
    print(f"Creating personal orgs for {len(users)} users")
    personal_orgs: dict[str, tuple[str, str]] = {}
    for user in users:
        personal_orgs[user.id] = await create_personal_org(db, user)

    print(f"Creating {NUM_SHARED_ORGS} shared orgs")
    # org_id -> {team_id -> [member user ids]}
    shared_org_teams: dict[str, dict[str, list[str]]] = {}
    # user_id -> [org ids the user is a member of]
    user_shared_orgs: dict[str, list[str]] = {}
    if len(users) > 1:
        for _ in range(NUM_SHARED_ORGS):
            owner = random.choice(users)
            member_pool = [u for u in users if u.id != owner.id]
            num_members = min(
                random.randint(MIN_SHARED_ORG_MEMBERS, MAX_SHARED_ORG_MEMBERS),
                len(member_pool),
            )
            members = random.sample(member_pool, k=num_members)
            org_id, member_ids, team_membership = await create_shared_org(
                db, owner, members
            )
            shared_org_teams[org_id] = team_membership
            for uid in member_ids:
                user_shared_orgs.setdefault(uid, []).append(org_id)

    def pick_tenancy(user_id: str) -> tuple[str, str]:
        """Pick ``(organization_id, team_id)`` for a resource owned by a user.

        Most resources land in the user's personal org; a random minority land
        in a shared org/team the user belongs to so shared-visibility data exists.
        """
        candidate_org_ids = user_shared_orgs.get(user_id, [])
        if candidate_org_ids and random.random() < SHARED_TENANCY_RATIO:
            org_id = random.choice(candidate_org_ids)
            member_teams = [
                team_id
                for team_id, member_ids in shared_org_teams[org_id].items()
                if user_id in member_ids
            ]
            if member_teams:
                return org_id, random.choice(member_teams)
        return personal_orgs[user_id]

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
            org_id, team_id = pick_tenancy(user.id)
            graph = await db.agentgraph.create(
                data=AgentGraphCreateInput(
                    name=faker.sentence(nb_words=3),
                    description=faker.text(max_nb_chars=200),
                    userId=user.id,
                    isActive=True,
                    organizationId=org_id,
                    teamId=team_id,
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
            org_id, team_id = pick_tenancy(user.id)
            preset = await db.agentpreset.create(
                data={
                    "name": faker.sentence(nb_words=3),
                    "description": faker.text(max_nb_chars=200),
                    "userId": user.id,
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "isActive": True,
                    "organizationId": org_id,
                    "teamId": team_id,
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
                # Empty (not None) — Creator view requires non-null avatar_url.
                avatarUrl="",
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

            org_id, team_id = pick_tenancy(user.id)
            library_agent = await db.libraryagent.create(
                data={
                    "userId": user.id,
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "creatorId": creator_profile.id if creator_profile else None,
                    "imageUrl": None,
                    "useGraphIsActiveVersion": random.choice([True, False]),
                    "isFavorite": random.choice([True, False]),
                    "isCreatedByUser": random.choice([True, False]),
                    "isArchived": random.choice([True, False]),
                    "isDeleted": random.choice([True, False]),
                    "organizationId": org_id,
                    "teamId": team_id,
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

            org_id, team_id = pick_tenancy(user.id)
            graph_execution_data.append(
                {
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "userId": user.id,
                    "executionStatus": prisma.enums.AgentExecutionStatus.COMPLETED,
                    "startedAt": faker.date_time_this_year(),
                    "agentPresetId": preset.id if preset else None,
                    "organizationId": org_id,
                    "teamId": team_id,
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
                "imageUrls": [],
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
            org_id, team_id = pick_tenancy(user.id)
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
                    organizationId=org_id,
                    teamId=team_id,
                )
            )

    # Insert APIKeys
    print(f"Inserting {NUM_USERS} api keys")
    for user in users:
        api_key = APIKeySmith().generate_key()
        org_id, team_id = pick_tenancy(user.id)
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
                "organizationId": org_id,
                "teamId": team_id,
            }
        )

    # Refresh materialized views
    print("Refreshing materialized views...")
    await db.execute_raw("SELECT refresh_store_materialized_views();")

    await db.disconnect()
    print("Test data creation completed successfully!")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Data seeding test requires a dedicated database; not for CI",
)
async def test_main_function_runs_without_errors():
    await main()


if __name__ == "__main__":
    asyncio.run(main())
