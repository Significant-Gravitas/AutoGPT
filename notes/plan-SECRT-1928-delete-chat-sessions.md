# Implementation Plan: SECRT-1928 - Delete Chat Sessions

**Ticket:** [SECRT-1928](https://linear.app/autogpt/issue/SECRT-1928)
**Author:** Otto
**Date:** 2026-02-14

## Summary

Add the ability for users to delete chat sessions from the CoPilot interface. The backend logic already exists (`delete_chat_session` in `model.py` and `db.py`), it just needs a route and frontend UI.

## Current State

### Backend (already exists)
- `backend/api/features/chat/db.py:delete_chat_session()` - DB deletion with user ownership validation
- `backend/api/features/chat/model.py:delete_chat_session()` - Handles cache cleanup and lock removal
- **Missing:** No DELETE route in `routes.py`

### Frontend
- `ChatSidebar.tsx` displays session list with no delete option
- `MobileDrawer.tsx` also needs delete option
- **Missing:** Delete button, confirmation dialog, API call

## Implementation

### Phase 1: Backend Route (15 min)

**File:** `autogpt_platform/backend/backend/api/features/chat/routes.py`

Add after line ~200 (after `create_session`):

```python
@router.delete(
    "/sessions/{session_id}",
    dependencies=[Security(auth.requires_user)],
    status_code=204,
)
async def delete_session(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> Response:
    """
    Delete a chat session.

    Permanently removes a chat session and all its messages.
    Only the owner can delete their sessions.

    Args:
        session_id: The session ID to delete.
        user_id: The authenticated user's ID.

    Returns:
        204 No Content on success.

    Raises:
        404: Session not found or not owned by user.
    """
    from .model import delete_chat_session
    
    deleted = await delete_chat_session(session_id, user_id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or access denied"
        )
    
    return Response(status_code=204)
```

**Add import at top:**
```python
from fastapi import Response  # add to existing imports
```

### Phase 2: Frontend API Hook (auto-generated)

After adding the backend route, run the OpenAPI generator to create the hook:
```bash
cd autogpt_platform/frontend
pnpm generate:api
```

This will generate `useDeleteV2Session` or similar.

### Phase 3: Frontend UI (30 min)

**File:** `autogpt_platform/frontend/src/app/(platform)/copilot/components/ChatSidebar/ChatSidebar.tsx`

1. Add imports:
```tsx
import { TrashIcon } from "@phosphor-icons/react";
import { useDeleteV2Session } from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
```

2. Add state and mutation hook inside component:
```tsx
const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
const queryClient = useQueryClient();

const { mutate: deleteSession, isPending: isDeleting } = useDeleteV2Session({
  mutation: {
    onSuccess: () => {
      // Invalidate sessions list to refetch
      queryClient.invalidateQueries({ queryKey: ['v2', 'sessions'] });
      // If we deleted the current session, clear selection
      if (sessionToDelete === sessionId) {
        setSessionId(null);
      }
      setSessionToDelete(null);
    },
    onError: (error) => {
      console.error("Failed to delete session:", error);
      setSessionToDelete(null);
    },
  },
});

function handleDeleteClick(e: React.MouseEvent, id: string) {
  e.stopPropagation(); // Prevent session selection
  setSessionToDelete(id);
}

function handleConfirmDelete() {
  if (sessionToDelete) {
    deleteSession({ sessionId: sessionToDelete });
  }
}
```

3. Add delete button to each session item (inside the session button, after the date):
```tsx
<button
  onClick={(e) => handleDeleteClick(e, session.id)}
  className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 
             p-1.5 rounded hover:bg-red-100 text-zinc-400 hover:text-red-600 transition-all"
  aria-label="Delete chat"
>
  <TrashIcon className="h-4 w-4" />
</button>
```

4. Add confirmation dialog before the closing `</Sidebar>`:
```tsx
<AlertDialog open={!!sessionToDelete} onOpenChange={(open) => !open && setSessionToDelete(null)}>
  <AlertDialogContent>
    <AlertDialogHeader>
      <AlertDialogTitle>Delete this chat?</AlertDialogTitle>
      <AlertDialogDescription>
        This will permanently delete this conversation and all its messages. This action cannot be undone.
      </AlertDialogDescription>
    </AlertDialogHeader>
    <AlertDialogFooter>
      <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
      <AlertDialogAction 
        onClick={handleConfirmDelete}
        disabled={isDeleting}
        className="bg-red-600 hover:bg-red-700"
      >
        {isDeleting ? "Deleting..." : "Delete"}
      </AlertDialogAction>
    </AlertDialogFooter>
  </AlertDialogContent>
</AlertDialog>
```

5. Update session button to have `group` class for hover effects:
```tsx
<button
  key={session.id}
  onClick={() => handleSelectSession(session.id)}
  className={cn(
    "group relative w-full rounded-lg px-3 py-2.5 pr-10 text-left transition-colors",  // added group, relative, pr-10
    ...
  )}
>
```

### Phase 4: Mobile Drawer Update (15 min)

**File:** `autogpt_platform/frontend/src/app/(platform)/copilot/components/MobileDrawer/MobileDrawer.tsx`

Similar pattern - add delete button with swipe-to-delete or long-press context menu for mobile UX.

## Testing

1. **Backend:** 
   - `DELETE /api/v2/chat/sessions/{id}` returns 204 for valid session
   - Returns 404 for invalid/unowned session
   - Verify messages are cascade deleted (Prisma handles this)

2. **Frontend:**
   - Hover shows delete icon
   - Click shows confirmation
   - Cancel dismisses dialog
   - Confirm deletes and refreshes list
   - Deleting current session clears selection

## Migration/Rollback

None required - additive change only.

## Verified ✅

1. **`delete_chat_session` return behavior:** Returns `bool` - `True` if deleted successfully, `False` otherwise (not owned/not found). No exception raised. The route code is correct.

2. **Query key for cache invalidation:** Should use `getGetV2ListSessionsQueryKey()` from the generated code, not hardcoded `['v2', 'sessions']`. Fix:
   ```tsx
   import { getGetV2ListSessionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
   // ...
   queryClient.invalidateQueries({ queryKey: getGetV2ListSessionsQueryKey() });
   ```

3. **`setSessionId` availability:** Available from `useQueryState("sessionId", parseAsString)` already in component scope.

## Open Questions

1. Should we support bulk delete? (Deferred - out of scope for this ticket)
2. Soft delete vs hard delete? (Hard delete per Zamil's request - "idc")

## Files Changed

- `autogpt_platform/backend/backend/api/features/chat/routes.py`
- `autogpt_platform/frontend/src/app/(platform)/copilot/components/ChatSidebar/ChatSidebar.tsx`
- `autogpt_platform/frontend/src/app/(platform)/copilot/components/MobileDrawer/MobileDrawer.tsx`
