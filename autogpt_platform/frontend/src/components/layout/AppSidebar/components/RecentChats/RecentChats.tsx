"use client";

import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";
import { ShareChatDialog } from "@/app/(platform)/copilot/sharing/ShareChatDialog";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { SidebarMenu } from "@/components/ui/sidebar";
import { CaretDownIcon } from "@phosphor-icons/react";
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
      {groupSessionsByDate(sessions).map((group) => (
        <Collapsible key={group.label} defaultOpen className="group/day">
          <CollapsibleTrigger className="flex w-full items-center gap-2 px-2 pb-1 pt-2 text-xs font-medium text-zinc-600">
            <span>{group.label}</span>
            <CaretDownIcon
              weight="bold"
              className="ease-[cubic-bezier(0.33,1,0.68,1)] ml-auto size-3.5 text-zinc-500 transition-transform duration-200 group-data-[state=open]/day:rotate-180 motion-reduce:transition-none"
            />
          </CollapsibleTrigger>
          <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
            <SidebarMenu>{group.sessions.map(renderItem)}</SidebarMenu>
          </CollapsibleContent>
        </Collapsible>
      ))}

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
