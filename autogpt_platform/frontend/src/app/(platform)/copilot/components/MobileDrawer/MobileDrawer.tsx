import { Button } from "@/components/atoms/Button/Button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { Button as ShadcnButton } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  MagnifyingGlass,
  PlusIcon,
  SpinnerGapIcon,
  X,
} from "@phosphor-icons/react";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useRef } from "react";
import { Drawer } from "vaul";
import { useCopilotChatRuntimeStore } from "../../copilotChatRegistry";
import { shouldShowSessionProcessingIndicator } from "../../sessionActivity";
import { useCopilotUIStore } from "../../store";
import { useSessionDeletion } from "../../useSessionDeletion";
import { useSessionList } from "../../useSessionList";
import { ChatSearchResults } from "../ChatSearchModal/ChatSearchResults";
import { useChatSearch } from "../ChatSearchModal/useChatSearch";
import { ChatSessionBlock } from "../ChatSessionBlock/ChatSessionBlock";
import { NotificationToggle } from "../ChatSidebar/components/NotificationToggle/NotificationToggle";
import { DeleteChatDialog } from "../DeleteChatDialog/DeleteChatDialog";
import { UsagePopover } from "../UsageLimits/UsagePopover/UsagePopover";

export function MobileDrawer() {
  const { isUserLoading, isLoggedIn } = useSupabase();
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [currentSessionId, setSessionId] = useQueryState(
    "sessionId",
    parseAsString,
  );
  const {
    completedSessionIDs,
    clearCompletedSession,
    isDrawerOpen,
    setDrawerOpen,
    isSearchOpen,
    setSearchOpen,
  } = useCopilotUIStore();
  const isChatSearchEnabled = useGetFlag(Flag.CHAT_SEARCH);
  const isSearchActive = isChatSearchEnabled && isSearchOpen;
  const sessionNeedsReload = useCopilotChatRuntimeStore(
    (state) => state.sessionNeedsReload,
  );

  const { sessions, isLoading, hasMore, isLoadingMore, loadMore } =
    useSessionList({ enabled: !isUserLoading && isLoggedIn });
  const {
    query,
    debouncedQuery,
    setQuery,
    results,
    highlightedIndex,
    setHighlightedIndex,
    highlightedResultRef,
  } = useChatSearch(sessions, isSearchOpen);

  const { sessionToDelete, isDeleting, confirmDelete, cancelDelete } =
    useSessionDeletion();

  useEffect(() => {
    if (!isSearchActive || !isDrawerOpen) return;
    window.setTimeout(() => searchInputRef.current?.focus(), 0);
  }, [isSearchActive, isDrawerOpen]);

  function handleDrawerOpenChange(open: boolean) {
    setDrawerOpen(open);
    if (!open) setSearchOpen(false);
  }

  function closeDrawer() {
    setDrawerOpen(false);
    setSearchOpen(false);
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
    setSearchOpen(false);
    closeDrawer();
  }

  function handleNewChat() {
    setSessionId(null);
    closeDrawer();
  }

  return (
    <>
      <Drawer.Root
        open={isDrawerOpen}
        onOpenChange={handleDrawerOpenChange}
        direction="left"
      >
        <Drawer.Portal>
          <Drawer.Overlay className="fixed inset-0 z-[60] bg-black/10 backdrop-blur-sm" />
          <Drawer.Content className="fixed left-0 top-0 z-[70] flex h-full w-80 flex-col border-r border-zinc-200 bg-zinc-50">
            <div className="shrink-0 border-b border-zinc-200 px-4 py-2">
              <div className="flex items-center justify-between">
                <Drawer.Title className="text-lg font-semibold text-zinc-800">
                  Your chats
                </Drawer.Title>
                <div className="flex items-center gap-1">
                  <UsagePopover />
                  <NotificationToggle />
                  {isChatSearchEnabled ? (
                    <ShadcnButton
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label={
                        isSearchOpen ? "Close search" : "Search chats"
                      }
                      onClick={() => setSearchOpen(!isSearchOpen)}
                      className="rounded-full text-zinc-600 hover:bg-zinc-100"
                    >
                      {isSearchOpen ? (
                        <X className="h-4 w-4" />
                      ) : (
                        <MagnifyingGlass className="h-4 w-4" />
                      )}
                    </ShadcnButton>
                  ) : null}
                  <Button
                    variant="icon"
                    size="icon"
                    aria-label="Close sessions"
                    onClick={closeDrawer}
                    className="ml-3"
                  >
                    <X width="1rem" height="1rem" />
                  </Button>
                </div>
              </div>
              {currentSessionId && !isSearchActive ? (
                <div className="mt-2">
                  <Button
                    variant="primary"
                    size="small"
                    onClick={handleNewChat}
                    className="w-full"
                    leftIcon={<PlusIcon width="1rem" height="1rem" />}
                  >
                    New Chat
                  </Button>
                </div>
              ) : null}
            </div>
            <div
              className={cn(
                "flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto px-3 py-3",
                scrollbarStyles,
              )}
            >
              {isSearchActive ? (
                <div className="flex min-h-0 flex-1 flex-col">
                  <div className="px-1 pb-3">
                    <div className="relative">
                      <MagnifyingGlass className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-400" />
                      <Input
                        ref={searchInputRef}
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        placeholder="Search chats..."
                        aria-label="Search chats"
                        autoComplete="off"
                        className="h-10 bg-white pl-9 text-sm"
                      />
                    </div>
                  </div>
                  <Separator className="mb-2" />
                  <div className="px-1 pb-2 text-xs font-medium uppercase tracking-wide text-zinc-400">
                    {debouncedQuery.trim() ? "Results" : "Recent chats"}
                  </div>
                  {results.length > 0 ? (
                    <ChatSearchResults
                      results={results}
                      query={debouncedQuery}
                      highlightedIndex={highlightedIndex}
                      highlightedResultRef={highlightedResultRef}
                      currentSessionId={currentSessionId}
                      completedSessionIDs={completedSessionIDs}
                      sessionNeedsReload={sessionNeedsReload}
                      onHighlight={setHighlightedIndex}
                      onSelect={(id) => {
                        handleSelectSession(id);
                        if (completedSessionIDs.has(id)) {
                          clearCompletedSession(id);
                        }
                      }}
                    />
                  ) : (
                    <p className="py-4 text-center text-sm text-neutral-500">
                      No chats found
                    </p>
                  )}
                </div>
              ) : isLoading ? (
                <div className="flex items-center justify-center py-4">
                  <SpinnerGapIcon className="h-5 w-5 animate-spin text-neutral-400" />
                </div>
              ) : sessions.length === 0 ? (
                <p className="py-4 text-center text-sm text-neutral-500">
                  No conversations yet
                </p>
              ) : (
                sessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => {
                      handleSelectSession(session.id);
                      if (completedSessionIDs.has(session.id)) {
                        clearCompletedSession(session.id);
                      }
                    }}
                    className={cn(
                      "w-full rounded-lg px-3 py-2.5 text-left transition-colors",
                      session.id === currentSessionId
                        ? "bg-zinc-100"
                        : "hover:bg-zinc-50",
                    )}
                  >
                    <ChatSessionBlock
                      title={session.title}
                      updatedAt={session.updated_at}
                      sourcePlatform={session.source_platform}
                      isActive={session.id === currentSessionId}
                      showProcessing={
                        !!session.is_processing &&
                        shouldShowSessionProcessingIndicator({
                          sessionId: session.id,
                          currentSessionId,
                          isProcessing: session.is_processing,
                          hasCompletedIndicator: completedSessionIDs.has(
                            session.id,
                          ),
                          needsReload: !!sessionNeedsReload[session.id],
                        })
                      }
                      showCompleted={
                        completedSessionIDs.has(session.id) &&
                        session.id !== currentSessionId
                      }
                    />
                  </button>
                ))
              )}
              {hasMore && (
                <Button
                  variant="ghost"
                  size="small"
                  onClick={() => loadMore()}
                  loading={isLoadingMore}
                  disabled={isLoadingMore}
                  className="mt-2 w-full justify-center text-neutral-500"
                >
                  {isLoadingMore ? "Loading…" : "Load older chats"}
                </Button>
              )}
            </div>
          </Drawer.Content>
        </Drawer.Portal>
      </Drawer.Root>
      <DeleteChatDialog
        session={sessionToDelete}
        isDeleting={isDeleting}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />
    </>
  );
}
