"use client";

import { DeleteChatDialog } from "@/app/(platform)/copilot/components/DeleteChatDialog/DeleteChatDialog";
import { ShareChatDialog } from "@/app/(platform)/copilot/sharing/ShareChatDialog";
import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import type { SessionGroupResponse } from "@/app/api/__generated__/models/sessionGroupResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { SidebarMenu } from "@/components/molecules/Sidebar/Sidebar";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { PinIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { ExpertChatGroup } from "./components/ExpertChatGroup/ExpertChatGroup";
import { NewExpertGroup } from "./components/NewExpertGroup/NewExpertGroup";
import { RecentChatItem } from "./components/RecentChatItem/RecentChatItem";
import { getNewChatHref, getNewExpertsWithoutChats } from "./helpers";
import { useRecentChats } from "./useRecentChats";

export function RecentChats() {
  const chatPinningEnabled = useGetFlag(Flag.CHAT_PINNING);

  const isLearningEnabled = useGetFlag(Flag.DREAM_SKILL_LEARNING_ENABLED);
  const { expertsById, activeExperts, activeExpertIds } = useExpertMap();
  // The week a new expert keeps their entry is measured from when the list
  // mounted: an expert who joins later counts as new, and one only ages out
  // on a later visit.
  const [mountedAt] = useState(() => Date.now());
  const {
    groups,
    pinnedSessions,
    isLoading,
    loadError,
    retryLoad,
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
  } = useRecentChats({ pinnedSeparately: chatPinningEnabled });

  if (isLoading) {
    return (
      <div className="flex justify-center py-4">
        <LoadingSpinner size="small" className="text-foreground-tertiary" />
      </div>
    );
  }

  // A failed read must not pass for an empty history.
  if (loadError) {
    return (
      <ErrorCard
        className="mt-2"
        context="your chats"
        responseError={{
          message: loadError instanceof Error ? loadError.message : undefined,
        }}
        onRetry={() => void retryLoad()}
      />
    );
  }

  const newExperts = getNewExpertsWithoutChats(
    activeExperts,
    groups.map((group) => group.expert_id),
    mountedAt,
  );

  if (!groups.length && !pinnedSessions.length && !newExperts.length) {
    return (
      <div className="px-2 py-2">
        <Text variant="body-tight" as="p" tone="muted">
          No conversations yet
        </Text>
      </div>
    );
  }

  function renderItem(session: (typeof pinnedSessions)[number]) {
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

  // Otto stays first; an expert waiting to be greeted comes next, ahead of
  // the experts the user already talks to, who follow their latest chat.
  const ottoGroups = groups.filter((group) => !group.expert_id);
  const expertGroups = groups.filter((group) => !!group.expert_id);

  function renderGroup(group: SessionGroupResponse) {
    const expertId = group.expert_id ?? null;
    const expert = expertId ? expertsById.get(expertId) : null;
    return (
      // Keyed by expert id so each group's open state and loaded pages
      // survive the grouped read's periodic refetch.
      <ExpertChatGroup
        key={expertId ?? "autopilot"}
        expertId={expertId}
        label={expertId ? (expert?.name ?? "Expert") : "Otto"}
        role={expertId ? expert?.jobTitle || expert?.role : "Head of AI"}
        avatarUrl={expert?.avatarUrl ?? null}
        color={expert?.color ?? null}
        newChatHref={getNewChatHref(expertId, activeExpertIds)}
        isAutopilot={!expertId}
        total={group.total}
        previewSessions={group.sessions}
        pinnedSeparately={chatPinningEnabled}
        activeSessionId={activeSessionId}
        renderItem={renderItem}
      />
    );
  }

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
        {ottoGroups.map(renderGroup)}
        {newExperts.map((expert) => (
          <NewExpertGroup key={`new-${expert.id}`} expert={expert} />
        ))}
        {expertGroups.map(renderGroup)}
      </div>

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
