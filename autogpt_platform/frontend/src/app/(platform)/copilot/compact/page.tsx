"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { cn } from "@/lib/utils";
import { parseAsString, useQueryState } from "nuqs";
import { usePlatformChrome } from "../../PlatformChrome/usePlatformChrome";
import { CopilotChatActionsProvider } from "../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CompactComposer } from "./components/CompactComposer/CompactComposer";
import { CompactHeader } from "./components/CompactHeader/CompactHeader";
import { CompactTranscript } from "./components/CompactTranscript/CompactTranscript";
import { useCompactCopilotPage } from "./useCompactCopilotPage";

export default function CompactCopilotPage() {
  const { isUserLoading, isLoggedIn } = useAuth();
  const { showNewLayout } = usePlatformChrome();
  const [sessionId] = useQueryState("sessionId", parseAsString);

  return (
    <div
      className={cn(
        "flex min-h-0 w-full flex-col bg-background",
        showNewLayout
          ? "h-svh max-lg:pt-16"
          : "h-[calc(100svh-60px-var(--preview-banner-height,0px))]",
      )}
    >
      {isUserLoading || !isLoggedIn ? (
        <div className="flex flex-1 items-center justify-center">
          <LoadingSpinner className="text-zinc-400" />
        </div>
      ) : (
        <CompactChat key={sessionId ?? "new"} />
      )}
    </div>
  );
}

function CompactChat() {
  const page = useCompactCopilotPage();

  return (
    <CopilotChatActionsProvider
      onSend={(message) => page.onSend(message)}
      onBackendTurn={page.onBackendTurn}
    >
      <CompactHeader
        agentStatus={page.agentStatus}
        agentActivity={page.agentActivity}
        sessionId={page.sessionId}
        sessions={page.sessions}
        onOpenSession={page.openSession}
        onOpenDetailedView={page.openDetailedView}
      />
      <CompactTranscript
        sessionId={page.sessionId}
        messages={page.messages}
        agentStatus={page.agentStatus}
        agentActivity={page.agentActivity}
        isStreaming={page.isStreaming}
        queuedMessages={page.queuedMessages}
      />
      <CompactComposer
        onSend={(text) => page.onSend(text)}
        onStop={page.onStop}
        isBusy={page.isBusy}
        disabled={page.isLoadingSession || page.isSending}
      />
    </CopilotChatActionsProvider>
  );
}
