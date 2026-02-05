# Implementation Plan: Fix edit_agent Creating New Agents Instead of Updating

**Ticket:** SECRT-1857  
**Author:** Otto  
**Date:** 2026-02-05

## Problem Statement

When using the CoPilot's `edit_agent` tool to modify an existing agent, instead of updating the existing agent (creating a new version), it creates an entirely new agent. This clutters the user's library with duplicate agents.

## Root Cause Analysis

In `autogpt_platform/backend/backend/api/features/chat/tools/agent_generator/core.py`, the `save_agent_to_library()` function has this flow:

```python
async def save_agent_to_library(
    agent_json: dict[str, Any], user_id: str, is_update: bool = False
) -> tuple[Graph, Any]:
    
    # ... graph conversion ...
    
    if is_update:
        # ✅ Correctly creates new version with same graph ID
        if graph.id:
            existing_versions = await get_graph_all_versions(graph.id, user_id)
            if existing_versions:
                latest_version = max(v.version for v in existing_versions)
                graph.version = latest_version + 1
                _reassign_node_ids(graph)

    created_graph = await create_graph(graph, user_id)

    # ❌ BUG: Always creates NEW library agent instead of updating existing one
    library_agents = await library_db.create_library_agent(
        graph=created_graph,
        user_id=user_id,
        ...
    )
```

The issue is that `create_library_agent()` always creates a **new** `LibraryAgent` database entry, even when we're just updating an existing agent to a new version.

## Solution

Use `update_agent_version_in_library()` (which already exists in `db.py`) when `is_update=True` and an existing library agent exists for this graph.

### Code Changes

**File:** `autogpt_platform/backend/backend/api/features/chat/tools/agent_generator/core.py`

**Modified function:** `save_agent_to_library()`

```python
async def save_agent_to_library(
    agent_json: dict[str, Any], user_id: str, is_update: bool = False
) -> tuple[Graph, Any]:
    """Save agent to database and user's library.

    Args:
        agent_json: Agent JSON dict
        user_id: User ID
        is_update: Whether this is an update to an existing agent

    Returns:
        Tuple of (created Graph, LibraryAgent)
    """
    # Populate user_id in AgentExecutorBlock nodes before conversion
    _populate_agent_executor_user_ids(agent_json, user_id)

    graph = json_to_graph(agent_json)

    existing_library_agent = None
    if is_update and graph.id:
        # Check if there's an existing library agent for this graph
        existing_library_agent = await library_db.get_library_agent_by_graph_id(
            user_id, graph.id
        )
        
        if existing_library_agent:
            # Get latest version and increment
            existing_versions = await get_graph_all_versions(graph.id, user_id)
            if existing_versions:
                latest_version = max(v.version for v in existing_versions)
                graph.version = latest_version + 1
                _reassign_node_ids(graph)
                logger.info(f"Updating agent {graph.id} to version {graph.version}")
    
    if not is_update or not existing_library_agent:
        # Creating new agent
        graph.id = str(uuid.uuid4())
        graph.version = 1
        _reassign_node_ids(graph)
        logger.info(f"Creating new agent with ID {graph.id}")

    created_graph = await create_graph(graph, user_id)

    if is_update and existing_library_agent:
        # Update existing library agent to point to new version
        updated_library_agent = await library_db.update_agent_version_in_library(
            user_id=user_id,
            agent_graph_id=created_graph.id,
            agent_graph_version=created_graph.version,
        )
        return created_graph, updated_library_agent
    else:
        # Create new library agent
        library_agents = await library_db.create_library_agent(
            graph=created_graph,
            user_id=user_id,
            sensitive_action_safe_mode=True,
            create_library_agents_for_sub_graphs=False,
        )
        return created_graph, library_agents[0]
```

## Testing Plan

1. **Unit test:** Add test case that verifies `save_agent_to_library(is_update=True)` updates existing library agent
2. **Manual test:** 
   - Create an agent via CoPilot
   - Edit it via `edit_agent` tool
   - Verify same library agent ID, incremented graph version
   - Verify no duplicate agents in library

## Risk Assessment

**Low risk:**
- Uses existing, tested `update_agent_version_in_library()` function
- Falls back to creating new agent if no existing library agent found
- No changes to database schema

## Alternative Considered

Could also add an `update` parameter to `create_library_agent()`, but using the existing `update_agent_version_in_library()` function is cleaner and follows existing patterns.

## Verified

### `get_library_agent_by_graph_id` exists ✅

```python
async def get_library_agent_by_graph_id(
    user_id: str,
    graph_id: str,
    graph_version: Optional[int] = None,
) -> library_model.LibraryAgent | None:
```

- Located in `autogpt_platform/backend/backend/api/features/library/db.py`
- Accepts `(user_id, graph_id)` as positional args (with optional `graph_version`)
- Returns `LibraryAgent` or `None` if not found
- Filters by `userId`, `agentGraphId`, and `isDeleted=False`

Ready to implement.
