"use client";

import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";
import { ShareChatDialog } from "@/app/(platform)/copilot/sharing/ShareChatDialog";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { groupSessionsByExpert } from "@/app/(platform)/copilot/useSessionList";
import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { SidebarMenu } from "@/components/ui/sidebar";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { cn } from "@/lib/utils";
import { RecentChatItem } from "./components/RecentChatItem/RecentChatItem";
import { groupSessionsByDate } from "./helpers";
import { useRecentChats } from "./useRecentChats";

export function RecentChats() {
  const chatSharingEnabled = useGetFlag(Flag.CHAT_SHARING);
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const { expertsById } = useExpertMap();
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

  const groups = isExpertsEnabled
    ? groupSessionsByExpert(sessions).map((group) => ({
        key: group.expertId ?? "autopilot",
        label: group.expertId
          ? (expertsById.get(group.expertId)?.name ?? "Expert")
          : "Autopilot",
        avatarUrl: group.expertId
          ? (expertsById.get(group.expertId)?.avatarUrl ?? null)
          : null,
        role: group.expertId
          ? (expertsById.get(group.expertId)?.role ?? null)
          : null,
        showAvatar: true,
        sessions: group.sessions,
      }))
    : groupSessionsByDate(sessions).map((group) => ({
        key: group.label,
        label: group.label,
        avatarUrl: null,
        role: null,
        showAvatar: false,
        sessions: group.sessions,
      }));

  return (
    <>
      <div className="mt-2 flex flex-col gap-4">
        {groups.map((group) => (
          <div key={group.key}>
            <div
              className={cn(
                "flex items-center px-2 pb-1.5",
                group.showAvatar
                  ? "gap-2 text-[13px] font-medium text-zinc-900"
                  : "gap-1.5 text-xs font-medium text-zinc-500",
              )}
            >
              {group.showAvatar && (
                <Avatar className="h-5 w-5">
                  {group.avatarUrl ? (
                    <AvatarImage src={group.avatarUrl} alt={group.label} />
                  ) : null}
                  <AvatarFallback className="text-[9px]">
                    {group.label}
                  </AvatarFallback>
                </Avatar>
              )}
              <span className="truncate">{group.label}</span>
              {group.role ? (
                <span
                  className={cn(
                    "shrink-0 rounded-full px-1.5 py-px text-[10px] font-medium",
                    getExpertAccent(group.role).pill,
                  )}
                >
                  {group.role}
                </span>
              ) : null}
            </div>
            {group.showAvatar ? (
              <div className="ml-[17px] border-l border-zinc-200 pl-1.5">
                <SidebarMenu>{group.sessions.map(renderItem)}</SidebarMenu>
              </div>
            ) : (
              <SidebarMenu>{group.sessions.map(renderItem)}</SidebarMenu>
            )}
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
