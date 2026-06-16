"use client";

import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";
import { ShareChatDialog } from "@/app/(platform)/copilot/sharing/ShareChatDialog";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { SidebarMenu } from "@/components/ui/sidebar";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { RecentChatItem } from "./components/RecentChatItem/RecentChatItem";
import { useRecentChats } from "./useRecentChats";

export function RecentChats() {
  const chatSharingEnabled = useGetFlag(Flag.CHAT_SHARING);
  const {
    sessions,
    isLoading,
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

  return (
    <>
      <SidebarMenu>
        {sessions.map((session) => (
          <RecentChatItem
            key={session.id}
            session={session}
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
        ))}
      </SidebarMenu>

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
