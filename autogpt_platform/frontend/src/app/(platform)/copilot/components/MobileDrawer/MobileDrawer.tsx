import { Button } from "@/components/atoms/Button/Button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/atoms/Separator/Separator";
import { Text } from "@/components/atoms/Text/Text";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
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
import {
  Cancel01Icon,
  Loading03Icon,
  PlusSignIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function MobileDrawer() {
  const { isUserLoading, isLoggedIn } = useAuth();
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
          <Drawer.Overlay className="fixed inset-0 z-60 bg-black/10 backdrop-blur-xs" />
          <Drawer.Content className="fixed top-0 left-0 z-70 flex h-full w-80 flex-col border-r border-zinc-200 bg-zinc-50">
            <div className="shrink-0 border-b border-zinc-200 px-4 py-2">
              <div className="flex items-center justify-between">
                <Drawer.Title className="text-lg font-semibold text-zinc-800">
                  Your chats
                </Drawer.Title>
                <div className="flex items-center gap-1">
                  <UsagePopover />
                  <NotificationToggle />
                  {isChatSearchEnabled ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label={
                        isSearchOpen ? "Close search" : "Search chats"
                      }
                      withTooltip={false}
                      onClick={() => setSearchOpen(!isSearchOpen)}
                      className="rounded-full text-zinc-600 hover:border-transparent hover:bg-zinc-100"
                    >
                      {isSearchOpen ? (
                        <Icon icon={Cancel01Icon} className="h-4 w-4" />
                      ) : (
                        <Icon icon={Search01Icon} className="h-4 w-4" />
                      )}
                    </Button>
                  ) : null}
                  <Button
                    variant="icon"
                    size="icon-lg"
                    aria-label="Close sessions"
                    onClick={closeDrawer}
                    className="ml-3"
                  >
                    <Icon icon={Cancel01Icon} width="1rem" height="1rem" />
                  </Button>
                </div>
              </div>
              {currentSessionId && !isSearchActive ? (
                <div className="mt-2">
                  <Button
                    variant="primary"
                    size="md"
                    onClick={handleNewChat}
                    className="w-full"
                    leftIcon={
                      <Icon icon={PlusSignIcon} width="1rem" height="1rem" />
                    }
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
                      <Icon
                        icon={Search01Icon}
                        className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-zinc-400"
                      />
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
                  <div className="px-1 pb-2 text-xs font-medium tracking-wide text-zinc-400 uppercase">
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
                    <Text
                      variant="body"
                      tone="muted"
                      className="py-4 text-center"
                    >
                      No chats found
                    </Text>
                  )}
                </div>
              ) : isLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Icon
                    icon={Loading03Icon}
                    className="h-5 w-5 animate-spin text-zinc-400"
                  />
                </div>
              ) : sessions.length === 0 ? (
                <Text variant="body" tone="muted" className="py-4 text-center">
                  No conversations yet
                </Text>
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
                  size="md"
                  onClick={() => loadMore()}
                  loading={isLoadingMore}
                  disabled={isLoadingMore}
                  className="mt-2 w-full justify-center text-muted-foreground"
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
