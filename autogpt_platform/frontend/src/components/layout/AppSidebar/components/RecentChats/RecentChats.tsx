"use client";

import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";
import { ShareChatDialog } from "@/app/(platform)/copilot/sharing/ShareChatDialog";
import { groupSessionsByExpert } from "@/app/(platform)/copilot/useSessionList";
import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { SidebarMenu } from "@/components/ui/sidebar";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { Icon } from "@/components/atoms/Icon/Icon";
import { PinIcon } from "@hugeicons/core-free-icons";
import { ExpertChatGroup } from "./components/ExpertChatGroup/ExpertChatGroup";
import { RecentChatItem } from "./components/RecentChatItem/RecentChatItem";
import { groupSessionsByDate } from "./helpers";
import { useRecentChats } from "./useRecentChats";

export function RecentChats() {
  const chatSharingEnabled = useGetFlag(Flag.CHAT_SHARING);
  const chatPinningEnabled = useGetFlag(Flag.CHAT_PINNING);
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const { expertsById } = useExpertMap();
  const {
    sessions,
    isLoading,
    hasMore,
    isLoadingMore,
    loadMore,
    activeSessionId,
    togglePin,
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
        chatPinningEnabled={chatPinningEnabled}
        onPin={togglePin}
        onRename={startRename}
        onExport={exportChat}
        onShare={setSharingSessionId}
        onDelete={requestDelete}
      />
    );
  }

  const pinnedSessions = chatPinningEnabled
    ? sessions.filter((session) => !!session.is_pinned)
    : [];
  const unpinnedSessions =
    pinnedSessions.length > 0
      ? sessions.filter((session) => !session.is_pinned)
      : sessions;

  return (
    <>
      <div className="mt-2 flex flex-col gap-4">
        {pinnedSessions.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 px-2 pb-1.5 text-xs font-medium text-zinc-500">
              <Icon icon={PinIcon} className="size-3.5" />
              <span className="truncate">Pinned</span>
            </div>
            <SidebarMenu>{pinnedSessions.map(renderItem)}</SidebarMenu>
          </div>
        )}
        {isExpertsEnabled
          ? groupSessionsByExpert(unpinnedSessions).map((group) => {
              const expert = group.expertId
                ? expertsById.get(group.expertId)
                : null;
              return (
                // Keyed by expert id so each group's collapse/reveal state
                // survives the session list's periodic refetch.
                <ExpertChatGroup
                  key={group.expertId ?? "autopilot"}
                  label={
                    group.expertId ? (expert?.name ?? "Expert") : "Autopilot"
                  }
                  avatarUrl={expert?.avatarUrl ?? null}
                  role={expert?.role ?? null}
                  sessions={group.sessions}
                  renderItem={renderItem}
                />
              );
            })
          : groupSessionsByDate(unpinnedSessions).map((group) => (
              <div key={group.label}>
                <div className="flex items-center gap-1.5 px-2 pb-1.5 text-xs font-medium text-zinc-500">
                  <span className="truncate">{group.label}</span>
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
