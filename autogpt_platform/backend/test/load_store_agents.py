"""
Load Store Agents Script

This script loads the exported store agents from the agents/ folder into the test database.
It creates:
- A user and profile for the 'autogpt' creator
- AgentGraph records from JSON files
- StoreListing and StoreListingVersion records from CSV metadata
- Approves agents that have is_available=true in the CSV

Usage:
    cd backend
    poetry run load-store-agents
    # Images served by the stack itself instead of production media on GCS
    # (see test/store_media_fixtures.py); the e2e stack loads them this way.
    poetry run load-store-agents --local-media
"""

import argparse
import asyncio
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from test import store_media_fixtures

import prisma.enums
from prisma import Json, Prisma
from prisma.types import (
    AgentBlockCreateInput,
    AgentGraphCreateInput,
    AgentNodeCreateInput,
    AgentNodeLinkCreateInput,
    ProfileCreateInput,
    StoreListingCreateInput,
    StoreListingVersionCreateInput,
    UserCreateInput,
)

# Path to agents folder (relative to backend directory)
AGENTS_DIR = Path(__file__).parent.parent / "agents"
CSV_FILE = AGENTS_DIR / "StoreAgent_rows.csv"

# User constants for the autogpt creator (test data, not production)
# Fixed uuid4 for idempotency - same user is reused across script runs
AUTOGPT_USER_ID = "79d96c73-e6f5-4656-a83a-185b41ee0d06"
AUTOGPT_EMAIL = "autogpt-test@agpt.co"
AUTOGPT_USERNAME = "autogpt"
AUTOGPT_AVATAR_URL = "https://storage.googleapis.com/agpt-prod-public-site-media/users/b3e41ea4-2f4c-4964-927c-fe682d857bad/images/4b5781a6-49e1-433c-9a75-65af1be5c02d.png"


async def initialize_blocks(db: Prisma) -> set[str]:
    """Initialize agent blocks in the database from the registered blocks.

    Returns a set of block IDs that exist in the database.
    """
    from backend.blocks import get_blocks

    print("  Initializing agent blocks...")
    blocks = get_blocks()
    created_count = 0
    block_ids = set()

    for block_cls in blocks.values():
        block = block_cls()
        block_ids.add(block.id)
        existing_block = await db.agentblock.find_first(
            where={"OR": [{"id": block.id}, {"name": block.name}]}
        )
        if not existing_block:
            await db.agentblock.create(
                data=AgentBlockCreateInput(
                    id=block.id,
                    name=block.name,
                    inputSchema=json.dumps(block.input_schema.jsonschema()),
                    outputSchema=json.dumps(block.output_schema.jsonschema()),
                )
            )
            created_count += 1
        elif block.id != existing_block.id or block.name != existing_block.name:
            await db.agentblock.update(
                where={"id": existing_block.id},
                data={
                    "id": block.id,
                    "name": block.name,
                    "inputSchema": json.dumps(block.input_schema.jsonschema()),
                    "outputSchema": json.dumps(block.output_schema.jsonschema()),
                },
            )

    print(f"  Initialized {len(blocks)} blocks ({created_count} new)")
    return block_ids


async def ensure_block_exists(
    db: Prisma, block_id: str, known_blocks: set[str]
) -> bool:
    """Ensure a block exists in the database, create a placeholder if needed.

    Returns True if the block exists (or was created), False otherwise.
    """
    if block_id in known_blocks:
        return True

    # Check if it already exists in the database
    existing = await db.agentblock.find_unique(where={"id": block_id})
    if existing:
        known_blocks.add(block_id)
        return True

    # Create a placeholder block
    print(f"    Creating placeholder block: {block_id}")
    try:
        await db.agentblock.create(
            data=AgentBlockCreateInput(
                id=block_id,
                name=f"Placeholder_{block_id[:8]}",
                inputSchema="{}",
                outputSchema="{}",
            )
        )
        known_blocks.add(block_id)
        return True
    except Exception as e:
        print(f"    Warning: Could not create placeholder block {block_id}: {e}")
        return False


def parse_image_urls(image_str: str) -> list[str]:
    """Parse the image URLs from CSV format like ["url1","url2"]."""
    if not image_str or image_str == "[]":
        return []
    try:
        return json.loads(image_str)
    except json.JSONDecodeError:
        return []


def parse_categories(categories_str: str) -> list[str]:
    """Parse categories from CSV format like ["cat1","cat2"]."""
    if not categories_str or categories_str == "[]":
        return []
    try:
        return json.loads(categories_str)
    except json.JSONDecodeError:
        return []


def sanitize_slug(slug: str) -> str:
    """Ensure slug only contains valid characters."""
    return re.sub(r"[^a-z0-9-]", "", slug.lower())


async def create_user_and_profile(
    db: Prisma, avatar_url: str = AUTOGPT_AVATAR_URL, *, local_media: bool = False
) -> None:
    """Create the autogpt user and profile if they don't exist. With
    `local_media`, an existing profile (from a load without it) is pointed at
    the local avatar too."""
    # Check if user exists
    existing_user = await db.user.find_unique(where={"id": AUTOGPT_USER_ID})
    if existing_user:
        print(f"User {AUTOGPT_USER_ID} already exists, skipping user creation")
    else:
        print(f"Creating user {AUTOGPT_USER_ID}")
        await db.user.create(
            data=UserCreateInput(
                id=AUTOGPT_USER_ID,
                email=AUTOGPT_EMAIL,
                name="AutoGPT",
                metadata=Json({}),
                integrations="",
            )
        )

    # Check if profile exists
    existing_profile = await db.profile.find_first(where={"userId": AUTOGPT_USER_ID})
    if existing_profile and local_media:
        print(
            f"Profile for user {AUTOGPT_USER_ID} exists, pointing its avatar at local media"
        )
        await db.profile.update(
            where={"id": existing_profile.id}, data={"avatarUrl": avatar_url}
        )
    elif existing_profile:
        print(
            f"Profile for user {AUTOGPT_USER_ID} already exists, skipping profile creation"
        )
    else:
        print(f"Creating profile for user {AUTOGPT_USER_ID}")
        await db.profile.create(
            data=ProfileCreateInput(
                userId=AUTOGPT_USER_ID,
                name="AutoGPT",
                username=AUTOGPT_USERNAME,
                description="Official AutoGPT agents and templates",
                links=["https://agpt.co"],
                avatarUrl=avatar_url,
            )
        )


async def load_csv_metadata() -> dict[str, dict]:
    """Load CSV metadata and return a dict keyed by storeListingVersionId."""
    metadata = {}
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            version_id = row["storeListingVersionId"]
            metadata[version_id] = {
                "listing_id": row["listing_id"],
                "store_listing_version_id": version_id,
                "slug": sanitize_slug(row["slug"]),
                "agent_name": row["agent_name"],
                "agent_video": row["agent_video"] if row["agent_video"] else None,
                "agent_image": parse_image_urls(row["agent_image"]),
                "featured": row["featured"].lower() == "true",
                "sub_heading": row["sub_heading"],
                "description": row["description"],
                "categories": parse_categories(row["categories"]),
                "use_for_onboarding": row["useForOnboarding"].lower() == "true",
                "is_available": row["is_available"].lower() == "true",
            }
    return metadata


async def load_agent_json(json_path: Path) -> dict:
    """Load and parse an agent JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def create_agent_graph(
    db: Prisma, agent_data: dict, known_blocks: set[str]
) -> tuple[str, int]:
    """Create an AgentGraph and its nodes/links from JSON data."""
    graph_id = agent_data["id"]
    version = agent_data.get("version", 1)

    # Check if graph already exists
    existing_graph = await db.agentgraph.find_unique(
        where={"graphVersionId": {"id": graph_id, "version": version}}
    )
    if existing_graph:
        print(f"  Graph {graph_id} v{version} already exists, skipping")
        return graph_id, version

    print(
        f"  Creating graph {graph_id} v{version}: {agent_data.get('name', 'Unnamed')}"
    )

    # Create the main graph
    await db.agentgraph.create(
        data=AgentGraphCreateInput(
            id=graph_id,
            version=version,
            name=agent_data.get("name"),
            description=agent_data.get("description"),
            instructions=agent_data.get("instructions"),
            recommendedScheduleCron=agent_data.get("recommended_schedule_cron"),
            isActive=agent_data.get("is_active", True),
            userId=AUTOGPT_USER_ID,
            forkedFromId=agent_data.get("forked_from_id"),
            forkedFromVersion=agent_data.get("forked_from_version"),
        )
    )

    # Create nodes
    nodes = agent_data.get("nodes", [])
    for node in nodes:
        block_id = node["block_id"]
        # Ensure the block exists (create placeholder if needed)
        block_exists = await ensure_block_exists(db, block_id, known_blocks)
        if not block_exists:
            print(
                f"    Skipping node {node['id']} - block {block_id} could not be created"
            )
            continue

        await db.agentnode.create(
            data=AgentNodeCreateInput(
                id=node["id"],
                agentBlockId=block_id,
                agentGraphId=graph_id,
                agentGraphVersion=version,
                constantInput=Json(node.get("input_default", {})),
                metadata=Json(node.get("metadata", {})),
            )
        )

    # Create links
    links = agent_data.get("links", [])
    for link in links:
        await db.agentnodelink.create(
            data=AgentNodeLinkCreateInput(
                id=link["id"],
                agentNodeSourceId=link["source_id"],
                agentNodeSinkId=link["sink_id"],
                sourceName=link["source_name"],
                sinkName=link["sink_name"],
                isStatic=link.get("is_static", False),
            )
        )

    # Handle sub_graphs recursively
    sub_graphs = agent_data.get("sub_graphs", [])
    for sub_graph in sub_graphs:
        await create_agent_graph(db, sub_graph, known_blocks)

    return graph_id, version


async def create_store_listing(
    db: Prisma,
    graph_id: str,
    graph_version: int,
    metadata: dict,
    *,
    local_media: bool = False,
) -> None:
    """Create StoreListing and StoreListingVersion for an agent. With
    `local_media`, an existing listing's version (from a load without it) is
    pointed at the local images in `metadata` too."""
    listing_id = metadata["listing_id"]
    version_id = metadata["store_listing_version_id"]

    # Check if listing already exists
    existing_listing = await db.storelisting.find_unique(where={"id": listing_id})
    if existing_listing and local_media:
        print(
            f"  Store listing {listing_id} exists, pointing its images at local media"
        )
        await db.storelistingversion.update_many(
            where={"id": version_id}, data={"imageUrls": metadata["agent_image"]}
        )
        return
    if existing_listing:
        print(f"  Store listing {listing_id} already exists, skipping")
        return

    print(f"  Creating store listing: {metadata['agent_name']}")

    # Determine if this should be approved
    is_approved = metadata["is_available"]
    submission_status = (
        prisma.enums.SubmissionStatus.APPROVED
        if is_approved
        else prisma.enums.SubmissionStatus.PENDING
    )

    # Create the store listing first (without activeVersionId - will update after)
    await db.storelisting.create(
        data=StoreListingCreateInput(
            id=listing_id,
            slug=metadata["slug"],
            agentGraphId=graph_id,
            owningUserId=AUTOGPT_USER_ID,
            hasApprovedVersion=is_approved,
            useForOnboarding=metadata["use_for_onboarding"],
        )
    )

    # Create the store listing version
    await db.storelistingversion.create(
        data=StoreListingVersionCreateInput(
            id=version_id,
            version=1,
            agentGraphId=graph_id,
            agentGraphVersion=graph_version,
            name=metadata["agent_name"],
            subHeading=metadata["sub_heading"],
            videoUrl=metadata["agent_video"],
            imageUrls=metadata["agent_image"],
            description=metadata["description"],
            categories=metadata["categories"],
            isFeatured=metadata["featured"],
            isAvailable=metadata["is_available"],
            submissionStatus=submission_status,
            submittedAt=datetime.now() if is_approved else None,
            reviewedAt=datetime.now() if is_approved else None,
            storeListingId=listing_id,
        )
    )

    # Update the store listing with the active version if approved
    if is_approved:
        await db.storelisting.update(
            where={"id": listing_id},
            data={"ActiveVersion": {"connect": {"id": version_id}}},
        )


async def use_local_images(csv_metadata: dict[str, dict]) -> None:
    """Point every listing at images saved to the stack's own media storage,
    as many as the CSV gives it, instead of the CSV's production URLs."""
    for metadata in csv_metadata.values():
        metadata["agent_image"] = await store_media_fixtures.store_listing_images(
            AUTOGPT_USER_ID, metadata["slug"], len(metadata["agent_image"])
        )


async def main(local_media: bool = False):
    """Main function to load all store agents."""
    print("=" * 60)
    print("Loading Store Agents into Test Database")
    print("=" * 60)

    db = Prisma()
    await db.connect()

    try:
        # Step 0: Initialize agent blocks
        print("\n[Step 0] Initializing agent blocks...")
        known_blocks = await initialize_blocks(db)

        # Step 1: Create user and profile
        print("\n[Step 1] Creating user and profile...")
        # Files are saved before any row points at them.
        avatar_url = (
            await store_media_fixtures.store_avatar(AUTOGPT_USER_ID)
            if local_media
            else AUTOGPT_AVATAR_URL
        )
        await create_user_and_profile(db, avatar_url, local_media=local_media)

        # Step 2: Load CSV metadata
        print("\n[Step 2] Loading CSV metadata...")
        csv_metadata = await load_csv_metadata()
        print(f"  Found {len(csv_metadata)} store listing entries in CSV")
        if local_media:
            await use_local_images(csv_metadata)
            saved = sum(len(m["agent_image"]) for m in csv_metadata.values())
            print(f"  Saved {saved} listing images and the avatar to local media")

        # Step 3: Find all JSON files and match with CSV
        print("\n[Step 3] Processing agent JSON files...")
        json_files = list(AGENTS_DIR.glob("agent_*.json"))
        print(f"  Found {len(json_files)} agent JSON files")

        # Build mapping from version_id to json file
        loaded_graphs = {}  # graph_id -> (graph_id, version)
        failed_agents = []

        for json_file in json_files:
            # Extract the version ID from filename (agent_<version_id>.json)
            version_id = json_file.stem.replace("agent_", "")

            if version_id not in csv_metadata:
                print(
                    f"  Warning: {json_file.name} not found in CSV metadata, skipping"
                )
                continue

            metadata = csv_metadata[version_id]
            agent_name = metadata["agent_name"]
            print(f"\nProcessing: {agent_name}")

            # Use a transaction per agent to prevent dangling resources.
            # Generous timeout: large multi-graph agents exceed the 5s default
            # when the DB is remote (e.g. seeding a preview env on Cloud SQL).
            try:
                async with db.tx(timeout=timedelta(minutes=5)) as tx:
                    # Load and create the agent graph
                    agent_data = await load_agent_json(json_file)
                    graph_id, graph_version = await create_agent_graph(
                        tx, agent_data, known_blocks
                    )
                    loaded_graphs[graph_id] = (graph_id, graph_version)

                    # Create store listing
                    await create_store_listing(
                        tx, graph_id, graph_version, metadata, local_media=local_media
                    )
            except Exception as e:
                print(f"  Error loading agent '{agent_name}': {e}")
                failed_agents.append(agent_name)
                continue

        # Step 4: Refresh materialized views
        print("\n[Step 4] Refreshing materialized views...")
        try:
            await db.execute_raw("SELECT refresh_store_materialized_views();")
            print("  Materialized views refreshed successfully")
        except Exception as e:
            print(f"  Warning: Could not refresh materialized views: {e}")

        print("\n" + "=" * 60)
        print(f"Successfully loaded {len(loaded_graphs)} agents")
        if failed_agents:
            print(
                f"Failed to load {len(failed_agents)} agents: {', '.join(failed_agents)}"
            )
        print("=" * 60)

    finally:
        await db.disconnect()


def run():
    """Entry point for poetry script."""
    parser = argparse.ArgumentParser(description="Load the official store agents.")
    parser.add_argument(
        "--local-media",
        action="store_true",
        help="Give the listings and their creator placeholder images saved to "
        "the stack's own media storage, not the CSV's production image URLs. "
        "Listings and the profile an earlier load created are pointed at them too.",
    )
    asyncio.run(main(local_media=parser.parse_args().local_media))


if __name__ == "__main__":
    run()
