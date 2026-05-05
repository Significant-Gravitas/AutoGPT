import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { cn } from "@/lib/utils";
import {
  CheckCircle,
  CircleNotch,
  PlusIcon,
  SpeakerHigh,
  SpeakerSlash,
  SpinnerGapIcon,
  X,
} from "@phosphor-icons/react";
import { parseAsString, useQueryState } from "nuqs";
import { Drawer } from "vaul";
import { useCopilotChatRuntimeStore } from "../../copilotChatRegistry";
import { shouldShowSessionProcessingIndicator } from "../../sessionActivity";
import { useCopilotUIStore } from "../../store";
import { useSessionDeletion } from "../../useSessionDeletion";
import { DeleteChatDialog } from "../DeleteChatDialog/DeleteChatDialog";

function formatDate(dateString: string) {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;

  const day = date.getDate();
  const ordinal =
    day % 10 === 1 && day !== 11
      ? "st"
      : day % 10 === 2 && day !== 12
        ? "nd"
        : day % 10 === 3 && day !== 13
          ? "rd"
          : "th";
  const month = date.toLocaleDateString("en-US", { month: "short" });
  const year = date.getFullYear();

  return `${day}${ordinal} ${month} ${year}`;
}

export function MobileDrawer() {
  const { isUserLoading, isLoggedIn } = useSupabase();
  const [currentSessionId, setSessionId] = useQueryState(
    "sessionId",
    parseAsString,
  );
  const {
    completedSessionIDs,
    clearCompletedSession,
    isSoundEnabled,
    toggleSound,
    isDrawerOpen,
    setDrawerOpen,
  } = useCopilotUIStore();
  const sessionNeedsReload = useCopilotChatRuntimeStore(
    (state) => state.sessionNeedsReload,
  );

  const { data: sessionsResponse, isLoading } = useGetV2ListSessions(
    { limit: 50 },
    { query: { enabled: !isUserLoading && isLoggedIn } },
  );
  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  const { sessionToDelete, isDeleting, confirmDelete, cancelDelete } =
    useSessionDeletion();

  function closeDrawer() {
    setDrawerOpen(false);
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
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
        onOpenChange={setDrawerOpen}
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
                  <button
                    onClick={toggleSound}
                    className="rounded p-1.5 text-zinc-400 transition-colors hover:text-zinc-600"
                    aria-label={
                      isSoundEnabled
                        ? "Disable notification sound"
                        : "Enable notification sound"
                    }
                  >
                    {isSoundEnabled ? (
                      <SpeakerHigh className="h-4 w-4" />
                    ) : (
                      <SpeakerSlash className="h-4 w-4" />
                    )}
                  </button>
                  <Button
                    variant="icon"
                    size="icon"
                    aria-label="Close sessions"
                    onClick={closeDrawer}
                  >
                    <X width="1rem" height="1rem" />
                  </Button>
                </div>
              </div>
              {currentSessionId ? (
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
              {isLoading ? (
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
                    <div className="flex min-w-0 max-w-full flex-col overflow-hidden">
                      <div className="flex min-w-0 max-w-full items-center gap-1.5">
                        <Text
                          variant="body"
                          className={cn(
                            "truncate font-normal",
                            session.id === currentSessionId
                              ? "text-zinc-600"
                              : "text-zinc-800",
                          )}
                        >
                          {session.title || "Untitled chat"}
                        </Text>
                        {session.is_processing &&
                          shouldShowSessionProcessingIndicator({
                            sessionId: session.id,
                            currentSessionId,
                            isProcessing: session.is_processing,
                            hasCompletedIndicator: completedSessionIDs.has(
                              session.id,
                            ),
                            needsReload: !!sessionNeedsReload[session.id],
                          }) && (
                            <CircleNotch
                              className="h-4 w-4 shrink-0 animate-spin text-zinc-400"
                              weight="bold"
                            />
                          )}
                        {completedSessionIDs.has(session.id) &&
                          session.id !== currentSessionId && (
                            <CheckCircle
                              className="h-4 w-4 shrink-0 text-green-500"
                              weight="fill"
                            />
                          )}
                      </div>
                      <Text variant="small" className="text-neutral-400">
                        {formatDate(session.updated_at)}
                      </Text>
                    </div>
                  </button>
                ))
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
