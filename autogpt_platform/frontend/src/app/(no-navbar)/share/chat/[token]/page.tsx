"use client";

import { useMemo } from "react";
import { useParams } from "next/navigation";
import { ArtifactPanel } from "@/app/(platform)/copilot/components/ArtifactPanel/ArtifactPanel";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { sharedChatFilePattern, sharedChatFileUrl } from "@/lib/share/routes";
import { useSharedChatPage } from "./useSharedChatPage";
import { SharedChatErrorState } from "./components/SharedChatErrorState";
import { SharedChatLoadingState } from "./components/SharedChatLoadingState";

export default function SharedChatPage() {
  const params = useParams();
  const token = params.token as string;
  const isMobile = useIsMobile();
  // Compute the per-token pattern once per token so the renderer can
  // recognise file URLs we built for this specific share without
  // loosening the workspace-file matcher used by the owner side.
  const filePattern = useMemo(() => sharedChatFilePattern(token), [token]);
  // URL builder for inline ``workspace://`` rewrites in assistant prose
  // (handled by ``resolveWorkspaceUrls`` / ``extractWorkspaceArtifacts``)
  // — same token-aware shape as the converter's builder.
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

  if (isLoading) {
    return <SharedChatLoadingState />;
  }

  if (isError || !session) {
    return <SharedChatErrorState reason={error} onRetry={retry} />;
  }

  return (
    // Split layout — mirrors the owner copilot page: messages flex-1 on
    // the left, ArtifactPanel docked on the right when open.  The
    // ``useCopilotUIStore.artifactPanel`` state drives the panel's
    // visibility; ``useIsMobile`` picks docked vs Sheet rendering.
    <div className="flex h-screen w-full flex-row overflow-hidden">
      <div className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-col px-4 py-6">
          <header className="mb-4 space-y-1">
            <h1 className="text-2xl font-semibold">
              {session.title || "Shared chat"}
            </h1>
            <p className="text-sm text-zinc-500">
              Shared on {new Date(session.created_at).toLocaleDateString()} ·
              view only
            </p>
          </header>

          <div className="mb-4 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            This is a public read-only view of a chat conversation. The person
            who shared it can revoke access at any time.
          </div>

          {hasMore && (
            <div className="mb-4 rounded-md border border-zinc-200 bg-zinc-50 px-4 py-2 text-xs text-zinc-600">
              Showing the most recent {uiMessages.length} messages of this
              conversation. Older history is not visible in this shared view.
            </div>
          )}

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

          <div className="mt-4 text-center text-xs text-zinc-400">
            Powered by AutoGPT Platform
          </div>
        </div>
      </div>

      {/* Docked on desktop, fullscreen Sheet on mobile — same split
          as the owner-side copilot page.  Driven by the artifact
          panel store; the card's ``openArtifact`` sets state. */}
      {isMobile ? <ArtifactPanel mobile /> : <ArtifactPanel />}
    </div>
  );
}
