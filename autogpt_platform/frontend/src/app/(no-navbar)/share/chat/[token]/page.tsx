"use client";

import { useMemo } from "react";
import Image from "next/image";
import Link from "next/link";
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

  if (isLoading) {
    return <SharedChatLoadingState />;
  }

  if (isError || !session) {
    return <SharedChatErrorState reason={error} onRetry={retry} />;
  }

  return (
    // Mirrors the owner copilot layout: flex-row root, chat column on
    // the left fills available width, ArtifactPanel docks on the right
    // when open.  Background matches the owner side so the viewer
    // doesn't look like a different surface.
    <div className="flex h-screen w-full flex-row overflow-hidden bg-[#f8f8f9]">
      <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#f8f8f9]">
        {/* Header strip — AutoGPT logo on the left, chat title in the
            middle, Read-only pill on the right.  All ``shrink-0`` so a
            long chat can't push them off-screen. */}
        <header className="flex shrink-0 items-center justify-between gap-4 border-b border-zinc-200 bg-white px-4 py-3">
          <Link href="/" className="inline-block shrink-0">
            <Image
              src="/autogpt-logo-light-bg.png"
              alt="AutoGPT"
              width={120}
              height={54}
              className="block h-7 w-auto"
              priority
            />
          </Link>
          <div className="min-w-0 flex-1 text-center">
            <h1 className="truncate text-sm font-semibold text-zinc-900">
              {session.title || "Shared chat"}
            </h1>
            <p className="truncate text-xs text-zinc-500">
              Shared {new Date(session.created_at).toLocaleDateString()} ·
              public read-only view
            </p>
          </div>
          <span className="hidden shrink-0 rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-medium text-amber-800 sm:inline-block">
            Read-only
          </span>
        </header>

        {hasMore && (
          <div className="shrink-0 border-b border-zinc-200 bg-zinc-50 px-4 py-2 text-center text-xs text-zinc-600">
            Showing the most recent {uiMessages.length} messages — older history
            is not visible in this shared view.
          </div>
        )}

        {/* The chat column — ``flex-1 min-h-0`` so it takes only the
            remaining space between header and footer.  No max-width
            cap: the viewer has no sidebar to balance the layout, so
            we let messages fill the page width.  Internal scrolling
            lives inside ChatMessagesContainer's Conversation — we
            must NOT add a second overflow on the wrappers, otherwise
            the page ends up with two stacked scrollbars. */}
        <div className="flex min-h-0 w-full flex-1 flex-col bg-[#f8f8f9] px-2 lg:px-0">
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
        </div>

        {/* Pinned footer — outside the scroll area so it never gets
            run over by a long chat.  Tiny, low-contrast. */}
        <footer className="shrink-0 border-t border-zinc-200 bg-white px-4 py-2 text-center text-[11px] text-zinc-400">
          Powered by AutoGPT Platform
        </footer>
      </div>

      {/* Docked on desktop, fullscreen Sheet on mobile — driven by the
          artifact panel store; ArtifactCard's openArtifact sets state. */}
      {isMobile ? <ArtifactPanel mobile /> : <ArtifactPanel />}
    </div>
  );
}
