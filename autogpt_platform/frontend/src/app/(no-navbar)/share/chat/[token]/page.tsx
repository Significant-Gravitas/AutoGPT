"use client";

import { useCallback, useMemo } from "react";
import { useParams } from "next/navigation";
import { ArtifactPanel } from "@/app/(platform)/copilot/components/ArtifactPanel/ArtifactPanel";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { sharedChatFilePattern, sharedChatFileUrl } from "@/lib/share/routes";
import { useSharedChatPage } from "./useSharedChatPage";
import { SharedChatErrorState } from "./components/SharedChatErrorState";
import { SharedChatLoadingState } from "./components/SharedChatLoadingState";

// Public viewer has no send capability; this satisfies the actions
// context so tool-call cards render without throwing.
async function noopSend() {}

export default function SharedChatPage() {
  const params = useParams();
  const token = params.token as string;
  const isMobile = useIsMobile();
  const filePattern = useMemo(() => sharedChatFilePattern(token), [token]);
  const fileUrlBuilder = useMemo(
    () => (fileId: string) => sharedChatFileUrl(token, fileId),
    [token],
  );

  const {
    session,
    uiMessages,
    turnStats,
    hasMore,
    isLoading,
    isError,
    error,
    retry,
  } = useSharedChatPage(token);

  // Build the execution_id → share_token lookup so
  // ExecutionStartedCard's "View Execution" CTA routes to the linked
  // execution's own share page (``/share/{token}``) instead of the
  // auth-gated /library/agents/... URL.
  const linkedExecutions = session?.linked_executions ?? [];
  const getExecutionShareToken = useCallback(
    (executionId: string) => {
      const match = linkedExecutions.find(
        (link) => link.execution_id === executionId,
      );
      return match?.share_token ?? null;
    },
    [linkedExecutions],
  );

  if (isLoading) {
    return <SharedChatLoadingState />;
  }

  if (isError || !session) {
    return <SharedChatErrorState reason={error} onRetry={retry} />;
  }

  return (
    // Fills the layout's flex-1 body.  Owner-style flex-row so
    // ArtifactPanel docks alongside chat on desktop.
    <div className="flex h-full w-full flex-row overflow-hidden bg-[#f8f8f9]">
      <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#f8f8f9]">
        <h1 className="sr-only">{session.title || "Shared chat"}</h1>
        {hasMore && (
          <div className="shrink-0 border-b border-zinc-200 bg-zinc-50 px-4 py-2 text-center text-xs text-zinc-600">
            Showing the most recent {uiMessages.length} messages — older history
            is not visible in this shared view.
          </div>
        )}
        <div className="flex min-h-0 w-full flex-1 flex-col bg-[#f8f8f9] px-2 lg:px-0">
          <CopilotChatActionsProvider
            onSend={noopSend}
            chatSurface="share"
            getExecutionShareToken={getExecutionShareToken}
          >
            <ChatMessagesContainer
              messages={uiMessages}
              status="idle"
              error={undefined}
              isLoading={false}
              turnStats={turnStats}
              readOnly
              filePattern={filePattern}
              fileUrlBuilder={fileUrlBuilder}
            />
          </CopilotChatActionsProvider>
        </div>
      </div>
      {isMobile ? <ArtifactPanel mobile /> : <ArtifactPanel />}
    </div>
  );
}
