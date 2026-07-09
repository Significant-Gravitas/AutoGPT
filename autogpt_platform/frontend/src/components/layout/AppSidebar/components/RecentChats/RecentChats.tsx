"use client";

import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";
import { ShareChatDialog } from "@/app/(platform)/copilot/sharing/ShareChatDialog";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { SidebarMenu } from "@/components/ui/sidebar";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { RecentChatItem } from "./components/RecentChatItem/RecentChatItem";
import { groupSessionsByDate } from "./helpers";
import { useRecentChats } from "./useRecentChats";

export function RecentChats() {
  const chatSharingEnabled = useGetFlag(Flag.CHAT_SHARING);
  const {
    sessions,
    isLoading,
    hasMore,
    isLoadingMore,
    loadMore,
    activeSessionId,
    editingSessionId,
    editingTitle,
    setEditingTitle,
    startRename,
    submitRename,
    cancelRename,
    exportingIds,
    exportChat,
    sharingSessionId,
    setSharingSessionId,
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  } = useRecentChats();

  if (isLoading) {
    return (
      <div className="flex justify-center py-4">
        <LoadingSpinner size="small" className="text-neutral-500" />
      </div>
    );
  }

  if (!sessions.length) {
    return (
      <p className="px-2 py-2 text-sm text-neutral-500">No conversations yet</p>
    );
  }

  function renderItem(session: (typeof sessions)[number]) {
    return (
      <RecentChatItem
        key={session.id}
        session={session}
        isActive={session.id === activeSessionId}
        isEditing={editingSessionId === session.id}
        editingTitle={editingTitle}
        onEditingTitleChange={setEditingTitle}
        onSubmitRename={submitRename}
        onCancelRename={cancelRename}
        isExporting={exportingIds.has(session.id)}
        isDeleting={isDeleting}
        chatSharingEnabled={chatSharingEnabled}
        onRename={startRename}
        onExport={exportChat}
        onShare={setSharingSessionId}
        onDelete={requestDelete}
      />
    );
  }

  return (
    <>
      <div className="mt-2 flex flex-col gap-4">
        {groupSessionsByDate(sessions).map((group) => (
          <div key={group.label}>
            <div className="px-2 pb-1 text-xs font-medium text-zinc-600">
              {group.label}
            </div>
            <SidebarMenu>{group.sessions.map(renderItem)}</SidebarMenu>
          </div>
        ))}
      </div>

      {hasMore && (
        <button
          type="button"
          onClick={() => loadMore()}
          disabled={isLoadingMore}
          className="mt-1 flex w-full items-center justify-center gap-2 rounded-md bg-zinc-200 px-2 py-1.5 text-sm text-zinc-800 hover:bg-zinc-300 disabled:opacity-60"
        >
          {isLoadingMore && (
            <LoadingSpinner size="small" className="size-4 text-zinc-500" />
          )}
          {isLoadingMore ? "Loading…" : "Load more"}
        </button>
      )}

      <DeleteChatDialog
        session={sessionToDelete}
        isDeleting={isDeleting}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />

      {sharingSessionId && (
        <ShareChatDialog
          sessionId={sharingSessionId}
          open={true}
          onOpenChange={(next) => {
            if (!next) setSharingSessionId(null);
          }}
        />
      )}
    </>
  );
}
