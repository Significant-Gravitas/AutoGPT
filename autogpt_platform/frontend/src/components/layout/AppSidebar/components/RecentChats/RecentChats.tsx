"use client";

import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";
import { ShareChatDialog } from "@/app/(platform)/copilot/sharing/ShareChatDialog";
import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { groupSessionsByExpert } from "@/app/(platform)/copilot/useSessionList";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { SidebarMenu } from "@/components/molecules/Sidebar/Sidebar";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { PinIcon } from "@hugeicons/core-free-icons";
import { ExpertChatGroup } from "./components/ExpertChatGroup/ExpertChatGroup";
import { RecentChatItem } from "./components/RecentChatItem/RecentChatItem";
import { getNewChatHref } from "./helpers";
import { useRecentChats } from "./useRecentChats";

export function RecentChats() {
  const chatPinningEnabled = useGetFlag(Flag.CHAT_PINNING);

  const isLearningEnabled = useGetFlag(Flag.DREAM_SKILL_LEARNING_ENABLED);
  const { expertsById, activeExpertIds } = useExpertMap();
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
    excludeFromLearning,
    sessionToDelete,
    isDeleting,
    requestDelete,
    confirmDelete,
    cancelDelete,
  } = useRecentChats();

  if (isLoading) {
    return (
      <div className="flex justify-center py-4">
        <LoadingSpinner
          size="small"
          // TODO(tokens): no exact role for text-neutral-500 (#737373)
          className="text-neutral-500"
        />
      </div>
    );
  }

  if (!sessions.length) {
    return (
      <Text
        variant="body-tight"
        as="p"
        // TODO(tokens): no exact role for text-neutral-500 (#737373)
        className="px-2 py-2 text-neutral-500"
      >
        No conversations yet
      </Text>
    );
  }

  function renderItem(session: (typeof sessions)[number]) {
    const isEditing = editingSessionId === session.id;
    return (
      <RecentChatItem
        key={session.id}
        session={session}
        isActive={session.id === activeSessionId}
        isEditing={isEditing}
        editingTitle={isEditing ? editingTitle : ""}
        onEditingTitleChange={setEditingTitle}
        onSubmitRename={submitRename}
        onCancelRename={cancelRename}
        isExporting={exportingIds.has(session.id)}
        isDeleting={isDeleting}
        chatPinningEnabled={chatPinningEnabled}
        onPin={togglePin}
        onRename={startRename}
        onExport={exportChat}
        onShare={setSharingSessionId}
        isLearningEnabled={isLearningEnabled}
        onExcludeFromLearning={excludeFromLearning}
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
            <div className="flex items-center gap-1.5 px-2 pb-1.5 text-foreground-tertiary">
              <Icon icon={PinIcon} className="size-3.5" />
              <Text
                variant="small-medium-tight"
                as="span"
                tone="muted"
                className="truncate"
              >
                Pinned
              </Text>
            </div>
            <SidebarMenu>{pinnedSessions.map(renderItem)}</SidebarMenu>
          </div>
        )}
        {groupSessionsByExpert(unpinnedSessions).map((group) => {
          const expert = group.expertId
            ? expertsById.get(group.expertId)
            : null;
          return (
            // Keyed by expert id so each group's collapse/reveal state
            // survives the session list's periodic refetch.
            <ExpertChatGroup
              key={group.expertId ?? "autopilot"}
              label={group.expertId ? (expert?.name ?? "Expert") : "Otto"}
              role={
                group.expertId ? expert?.jobTitle || expert?.role : "Head of AI"
              }
              avatarUrl={expert?.avatarUrl ?? null}
              color={expert?.color ?? null}
              newChatHref={getNewChatHref(group.expertId, activeExpertIds)}
              isAutopilot={!group.expertId}
              sessions={group.sessions}
              renderItem={renderItem}
            />
          );
        })}
      </div>

      {hasMore && (
        <button
          type="button"
          onClick={() => loadMore()}
          disabled={isLoadingMore}
          // TODO(tokens): no fitting role for hover:bg-zinc-300 #C5C5C9; #C5C5C9 is foreground-disabled (a text colour) and border-strong (a border colour), not a fill
          className="mt-1 flex w-full items-center justify-center gap-2 rounded-md bg-surface-active px-2 py-1.5 text-foreground-emphasis hover:bg-zinc-300 disabled:opacity-60 max-md:py-3"
        >
          {isLoadingMore && (
            <LoadingSpinner
              size="small"
              className="size-4 text-foreground-tertiary"
            />
          )}
          <Text as="span" variant="body-tight-touch" tone="inherit">
            {isLoadingMore ? "Loading…" : "Load more"}
          </Text>
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
