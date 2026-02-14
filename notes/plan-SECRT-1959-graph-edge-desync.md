# Implementation Plan: SECRT-1959 - Agent Builder Graph Desync (Edge Deletion Not Persisting)

**Ticket:** [SECRT-1959](https://linear.app/autogpt/issue/SECRT-1959)
**Author:** Otto
**Date:** 2026-02-14
**Priority:** Urgent

## Problem Statement

Edge deletions in the Agent Builder are not persisting correctly. When users delete edges between nodes, the deletion doesn't stick — causing stale/orphan node connections across builds that cannot be corrected.

**Impact:**
- Users CANNOT delete edge endpoints to fix corrupt builds
- Workarounds (duplicating agents) are wasteful
- Agents end up with broken wiring that can't be repaired

## Root Cause Analysis

### Current Flow
1. User deletes edge → `edgeStore.removeEdge()` removes from Zustand state
2. User saves → `useSaveGraph` calls `getBackendLinks()` to get current edges
3. Backend creates new graph version with provided nodes/links
4. On reload, `useFlow` loads `graph.links` from backend into edge store

### Potential Failure Points

1. **Frontend state not syncing properly** - Edge removal might not be captured when save happens
2. **Draft manager interference** - `useDraftManager.ts` manages local drafts, might restore old state
3. **React Query cache** - Stale data might be served after save
4. **Backend not returning authoritative state** - Frontend might not update with server response

## Investigation Steps (Before Implementation)

**Need to verify:**
1. Is `getBackendLinks()` returning correct edges at save time?
2. Is the backend receiving the correct links in the update request?
3. Is the backend returning the new graph version with correct links?
4. Is the frontend updating from the backend response?

Add debug logging to trace the flow:

```typescript
// In useSaveGraph.ts, before save:
console.log('[SAVE] Links being sent:', useEdgeStore.getState().getBackendLinks());

// After save response:
console.log('[SAVE] Links returned:', response.data.links);
```

## Proposed Fix

### Phase 1: Frontend - Ensure State Consistency (Primary Fix)

**File:** `autogpt_platform/frontend/src/app/(platform)/build/hooks/useSaveGraph.ts`

After save completes, sync the edge store with the authoritative backend state:

```typescript
// In onSuccess callback after save
const { mutateAsync: updateGraph, isPending: isUpdating } =
  usePutV1UpdateGraphVersion({
    mutation: {
      onSuccess: async (response) => {
        const data = response.data as GraphModel;
        
        // ... existing code ...
        
        // CRITICAL: Sync edge store with authoritative backend state
        if (data.links) {
          useEdgeStore.getState().setEdges([]);
          useEdgeStore.getState().addLinks(data.links);
        }
        
        onSuccess?.(data);
        // ... rest of handler
      },
    },
  });
```

Same fix needed in `createNewGraph` mutation handler.

### Phase 2: Backend - Add Orphan Edge Validation (Defense in Depth)

**File:** `autogpt_platform/backend/backend/data/graph.py`

In `Graph.validate_graph()` or a new method, add orphan edge pruning:

```python
def normalize_links(self) -> list[Link]:
    """Remove orphan links that reference non-existent nodes."""
    valid_node_ids = {node.id for node in self.nodes}
    original_count = len(self.links)
    
    valid_links = [
        link for link in self.links
        if link.source_id in valid_node_ids and link.sink_id in valid_node_ids
    ]
    
    pruned_count = original_count - len(valid_links)
    if pruned_count > 0:
        logger.warning(
            f"Pruned {pruned_count} orphan link(s) from graph {self.id}"
        )
    
    return valid_links
```

Call this in `validate_graph()` or in `__create_graph()` before persisting:

```python
async def __create_graph(tx, graph: Graph, user_id: str):
    # Normalize links before saving
    graph.links = graph.normalize_links()
    
    graphs = [graph] + graph.sub_graphs
    # ... rest of function
```

### Phase 3: Frontend - Invalidate React Query Cache (Belt and Suspenders)

**File:** `autogpt_platform/frontend/src/app/(platform)/build/hooks/useSaveGraph.ts`

Ensure React Query refetches fresh data:

```typescript
import { useQueryClient } from "@tanstack/react-query";

// In the hook:
const queryClient = useQueryClient();

// In onSuccess:
onSuccess: async (response) => {
  const data = response.data as GraphModel;
  
  // Invalidate and refetch the graph query
  await queryClient.invalidateQueries({
    queryKey: ['v1', 'graphs', data.id],
  });
  
  // ... rest of handler
}
```

### Phase 4: Draft Manager Safety (Prevent Stale Restore)

**File:** `autogpt_platform/frontend/src/app/(platform)/build/components/FlowEditor/Flow/useDraftManager.ts`

Review draft restore logic to ensure it doesn't restore stale edge state after a save. The draft should be invalidated after a successful save (already happens in `useSaveGraph.ts`):

```typescript
// After save success in useSaveGraph.ts (already exists):
if (data.id) {
  await draftService.deleteDraft(data.id);
}
```

Verify this is working correctly.

## Testing Plan

### Manual Testing
1. Create agent with 3+ nodes and edges
2. Delete an edge
3. Save the graph
4. Refresh the page
5. Verify edge is still deleted
6. Check different scenarios:
   - Delete edge, save, close tab, reopen
   - Delete edge, save, navigate away, navigate back
   - Delete multiple edges, save

### Automated Testing
Add integration test:

```typescript
// In a test file
it('should persist edge deletion after save and reload', async () => {
  // Create graph with edge
  // Delete edge from store
  // Save graph
  // Clear stores
  // Reload graph
  // Assert edge is not present
});
```

## Rollback Plan

All changes are additive/defensive. If issues arise:
1. Revert the edge store sync in `useSaveGraph.ts`
2. Revert backend normalization (optional, only affects orphan cleanup)

## Files Changed

**Phase 1 (Critical):**
- `autogpt_platform/frontend/src/app/(platform)/build/hooks/useSaveGraph.ts`

**Phase 2 (Defense):**
- `autogpt_platform/backend/backend/data/graph.py`

**Phase 3 (Optional):**
- `autogpt_platform/frontend/src/app/(platform)/build/hooks/useSaveGraph.ts` (query invalidation)

## Open Questions

1. **Is there a way to reproduce this consistently?** Need more info from Zamil about exact repro steps
2. **Is AutoPilot creating agents differently?** The first screenshot suggests AutoPilot-created agents might have this issue more frequently
3. **Are there existing graphs with orphan edges in prod?** May need a migration script to clean up

## Estimation

- Phase 1 (Frontend sync): 30 min
- Phase 2 (Backend validation): 30 min  
- Phase 3 (Cache invalidation): 15 min
- Testing: 1 hour
- **Total: ~2-3 hours**

## Notes

The core issue is likely that the frontend isn't syncing with the authoritative backend state after save. Phase 1 is the critical fix. Phases 2-3 are defensive measures.
