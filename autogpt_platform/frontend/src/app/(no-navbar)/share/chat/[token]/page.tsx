"use client";

import { useCallback, useMemo, type ReactNode } from "react";
import dynamic from "next/dynamic";
import { useParams } from "next/navigation";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { sharedChatFilePattern, sharedChatFileUrl } from "@/lib/share/routes";
import { ShareActions } from "../../components/ShareHeader/ShareActions";
import { ShareHeader } from "../../components/ShareHeader/ShareHeader";
import { useSharedChatPage } from "./useSharedChatPage";
import { SharedChatErrorState } from "./components/SharedChatErrorState";
import { SharedChatLoadingState } from "./components/SharedChatLoadingState";

// Wraps every chat-share state — loading, error, success — in the
// branded shell so a viewer never sees a raw card without the logo
// + CTAs.  Matches ``ExecutionShareChrome`` on the execution share
// page so the two routes feel like the same surface.
//
// Uses ``bg-background`` (theme-aware) on the chrome wrappers so dark
// mode renders correctly.  Inner success-state wrappers still use the
// owner-side copilot's hardcoded ``#f8f8f9`` so the in-chat surface
// matches the owner experience pixel-for-pixel — see the
// ``CopilotPage`` / ``ChatContainer`` styling.
function SharedChatChrome({
  title,
  subtitle,
  children,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex h-screen w-full flex-col bg-background">
      <ShareHeader
        title={title}
        subtitle={subtitle}
        actions={<ShareActions />}
      />
      <div className="flex min-h-0 w-full flex-1 flex-row overflow-hidden bg-background">
        {children}
      </div>
    </div>
  );
}

// Dynamic import keeps the artifact panel + its viewer dependencies
// out of the initial bundle for public viewers who may never open
// one.  Mirrors the owner CopilotPage.
const ArtifactPanel = dynamic(
  () =>
    import(
      "@/app/(platform)/copilot/components/ArtifactPanel/ArtifactPanel"
    ).then((m) => m.ArtifactPanel),
  { ssr: false },
);

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

  // execution_id → public share_token, so ExecutionStartedCard's
  // "View Execution" CTA routes to /share/{token} instead of the
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
    return (
      <SharedChatChrome>
        <SharedChatLoadingState />
      </SharedChatChrome>
    );
  }

  if (isError || !session) {
    return (
      <SharedChatChrome>
        <SharedChatErrorState reason={error} onRetry={retry} />
      </SharedChatChrome>
    );
  }

  const title = session.title || "Shared chat";
  // ``shared_at`` is when the owner enabled sharing — that's what
  // belongs in the "Shared {date}" subtitle.  Falling back to
  // ``created_at`` only when ``shared_at`` is missing keeps already-
  // shared chats rendering until the backfill lands, but
  // ``SharedChatSession`` ships ``shared_at`` for every share so the
  // fallback should never fire in practice.
  const sharedOn = new Date(
    session.shared_at || session.created_at,
  ).toLocaleDateString();

  return (
    <SharedChatChrome title={title} subtitle={`Shared ${sharedOn}`}>
      <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#f8f8f9]">
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
    </SharedChatChrome>
  );
}
