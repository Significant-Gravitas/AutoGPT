"use client";

import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";
import { ShareChatDialog } from "@/app/(platform)/copilot/sharing/ShareChatDialog";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { SidebarMenu } from "@/components/ui/sidebar";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { CalendarBlankIcon, ListIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { RecentChatItem } from "./components/RecentChatItem/RecentChatItem";
import { groupSessionsByDate } from "./helpers";
import { useRecentChats } from "./useRecentChats";

// TEMPORARY: toggle between two date-display variants while the team decides
// which one to keep. Remove this along with the toggle button once chosen.
type DateView = "grouped" | "inline";

export function RecentChats() {
  const chatSharingEnabled = useGetFlag(Flag.CHAT_SHARING);
  const [dateView, setDateView] = useState<DateView>("grouped");
  const {
    sessions,
    isLoading,
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

  function renderItem(session: (typeof sessions)[number], showDate: boolean) {
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
        showDate={showDate}
        onRename={startRename}
        onExport={exportChat}
        onShare={setSharingSessionId}
        onDelete={requestDelete}
      />
    );
  }

  return (
    <>
      {/* TEMPORARY: variant toggle for team review — remove once decided. */}
      <button
        type="button"
        onClick={() =>
          setDateView((prev) => (prev === "grouped" ? "inline" : "grouped"))
        }
        className="mb-1 flex w-full items-center gap-1.5 rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-200"
      >
        {dateView === "grouped" ? (
          <ListIcon className="size-3.5" />
        ) : (
          <CalendarBlankIcon className="size-3.5" />
        )}
        {dateView === "grouped" ? "Show date inline" : "Group by date"}
      </button>

      {dateView === "grouped" ? (
        groupSessionsByDate(sessions).map((group) => (
          <div key={group.label}>
            <p className="px-2 pb-1 pt-2 text-xs font-medium text-zinc-400">
              {group.label}
            </p>
            <SidebarMenu>
              {group.sessions.map((session) => renderItem(session, false))}
            </SidebarMenu>
          </div>
        ))
      ) : (
        <SidebarMenu>
          {sessions.map((session) => renderItem(session, true))}
        </SidebarMenu>
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
