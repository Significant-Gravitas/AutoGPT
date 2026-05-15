import { toast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import type { UIMessage } from "ai";
import { useMemo, useRef } from "react";
import { concatWithAssistantMerge } from "./helpers/convertChatSessionToUiMessages";
import { getLatestAssistantStatusMessage } from "./helpers";
import { queueFollowUpMessage } from "./helpers/queueFollowUpMessage";
import { stripReplayPrefix } from "./helpers/stripReplayPrefix";
import { useCopilotStreamStore } from "./copilotStreamStore";
import { useCopilotPendingChips } from "./useCopilotPendingChips";
import { useCopilotUIStore } from "./store";
import { useChatSession } from "./useChatSession";
import { useCopilotNotifications } from "./useCopilotNotifications";
import { useCopilotStream } from "./useCopilotStream";
import { useLoadMoreMessages } from "./useLoadMoreMessages";
import { useSendMessage } from "./useSendMessage";
import { useSessionTitlePoll } from "./useSessionTitlePoll";
import { useWorkflowImportAutoSubmit } from "./useWorkflowImportAutoSubmit";

function trimVisibleMessagesForActiveRestore(messages: UIMessage[]) {
  const lastUserIndex = messages.findLastIndex(
    (message) => message.role === "user",
  );
  if (lastUserIndex === -1 || lastUserIndex === messages.length - 1) {
    return messages;
  }
  return messages.slice(0, lastUserIndex + 1);
}

function hasAssistantTail(messages: UIMessage[]) {
  const lastUserIndex = messages.findLastIndex(
    (message) => message.role === "user",
  );
  return lastUserIndex !== -1 && lastUserIndex < messages.length - 1;
}

export function useCopilotPage() {
  const { isUserLoading, isLoggedIn } = useSupabase();
  const isModeToggleEnabled = useGetFlag(Flag.CHAT_MODE_OPTION);

  const { copilotChatMode, copilotLlmModel, isDryRun } = useCopilotUIStore();

  const {
    sessionId,
    hydratedMessages,
    rawSessionMessages,
    historicalTurnStats,
    hasActiveStream,
    activeStreamStartedAt,
    hasMoreMessages,
    oldestSequence,
    isLoadingSession,
    isSessionError,
    createSession,
    isCreatingSession,
    refetchSession,
    sessionDryRun,
    sessionChatStatus,
  } = useChatSession({ dryRun: isDryRun });

  const {
    messages: currentMessages,
    setMessages,
    sendMessage,
    stop,
    status,
    error,
    isReconnecting,
    isRestoringActiveSession,
    isUserStoppingRef,
    isUserStopping,
    rateLimitMessage,
    dismissRateLimit,
  } = useCopilotStream({
    sessionId,
    hydratedMessages,
    hasActiveStream,
    refetchSession,
    copilotMode: isModeToggleEnabled ? copilotChatMode : undefined,
    copilotModel: isModeToggleEnabled ? copilotLlmModel : undefined,
  });

  const { pagedMessages, pagedTurnStats, hasMore, isLoadingMore, loadMore } =
    useLoadMoreMessages({
      sessionId,
      initialOldestSequence: oldestSequence,
      initialHasMore: hasMoreMessages,
      initialPageRawMessages: rawSessionMessages,
    });

  // Merge the older-pages and current-page stat maps; current-page (historical)
  // wins on overlap since it was persisted more recently.
  const turnStats = useMemo(() => {
    const merged = new Map(pagedTurnStats);
    historicalTurnStats?.forEach((v, k) => merged.set(k, v));
    return merged;
  }, [pagedTurnStats, historicalTurnStats]);

  // Ref that mirrors whether a stream turn is currently in-flight.
  // Updated synchronously on every render so it always reflects the latest
  // status — unlike reading `status` inside onSend (which captures the
  // closure's render-cycle value and can be stale for a frame).
  // Setting it to true *before* calling sendMessage prevents rapid
  // double-presses from both routing to /stream before React can re-render
  // with status="submitted".
  const isInflightRef = useRef(false);
  isInflightRef.current =
    !isUserStopping && (status === "streaming" || status === "submitted");

  // Combine paginated messages with current page messages, merging consecutive
  // assistant UIMessages at the page boundary so reasoning + response parts
  // stay in a single bubble. Paged messages are older history prepended before
  // the current page.
  const rawMessages = concatWithAssistantMerge(pagedMessages, currentMessages);
  const cachedSessionMessages = useMemo(
    () =>
      sessionId
        ? useCopilotStreamStore.getState().getMessageSnapshot(sessionId)
        : [],
    [sessionId],
  );
  const cachedRawMessages = concatWithAssistantMerge(
    pagedMessages,
    cachedSessionMessages,
  );

  // Drop / trim assistant messages whose leading text is a replay of an
  // earlier assistant (Claude Agent SDK's `--resume` behaviour). See
  // helpers/stripReplayPrefix.ts for the three cases.
  const messages = useMemo(() => stripReplayPrefix(rawMessages), [rawMessages]);
  const cachedMessages = useMemo(
    () => stripReplayPrefix(cachedRawMessages),
    [cachedRawMessages],
  );
  const restoreStatusMessage = useMemo(
    () =>
      isRestoringActiveSession
        ? getLatestAssistantStatusMessage(messages)
        : null,
    [isRestoringActiveSession, messages],
  );
  const displayMessages = useMemo(() => {
    if (!isRestoringActiveSession) return messages;
    if (hasAssistantTail(cachedMessages)) return cachedMessages;
    return trimVisibleMessagesForActiveRestore(messages);
  }, [isRestoringActiveSession, messages, cachedMessages]);

  // Chip state machine (peek sync + auto-continue promotion + mid-turn poll)
  // lives in a dedicated hook so this component is just glue.
  const { queuedMessages, queueMessage } = useCopilotPendingChips({
    sessionId,
    status,
    messages,
    setMessages,
  });

  useCopilotNotifications(sessionId);

  const {
    onSend: sendNewMessage,
    isUploadingFiles,
    setPendingFileParts,
  } = useSendMessage({
    sessionId,
    sendMessage,
    createSession,
    isUserStoppingRef,
  });

  // Wrap sendNewMessage with queue-in-flight routing: if a session is active
  // and a turn is already running, POST the follow-up text to the pending
  // endpoint so the backend buffers it; otherwise fall through to normal send.
  async function onSend(message: string, files?: File[]) {
    const trimmed = message.trim();
    if (!trimmed && (!files || files.length === 0)) return;

    if (sessionId && isInflightRef.current) {
      if (files && files.length > 0) {
        toast({
          title: "Please wait to attach files",
          description:
            "File attachments can't be queued until the current response finishes.",
          variant: "destructive",
        });
        return;
      }

      try {
        await queueFollowUpMessage(sessionId, trimmed);
        queueMessage(trimmed);
      } catch (err) {
        if (
          err instanceof Error &&
          err.name === "QueueFollowUpNotActiveError"
        ) {
          await sendNewMessage(message, files);
          return;
        }
        toast({
          title: "Could not queue message",
          description: "Please wait for the current response to finish.",
          variant: "destructive",
        });
        throw err;
      }
      return;
    }

    // Mark in-flight synchronously before dispatching so a rapid second
    // press sees isInflightRef.current=true and routes to the queue path
    // instead of triggering a duplicate /stream POST.
    if (sessionId) {
      isInflightRef.current = true;
    }
    await sendNewMessage(message, files);
  }

  useWorkflowImportAutoSubmit({ onSend, setPendingFileParts });

  useSessionTitlePoll({ sessionId, status, isReconnecting });

  return {
    sessionId,
    messages: displayMessages,
    status,
    error,
    stop,
    isReconnecting,
    isRestoringActiveSession,
    restoreStatusMessage,
    activeStreamStartedAt,
    isUserStopping,
    isLoadingSession,
    isSessionError,
    isCreatingSession,
    isUploadingFiles,
    isUserLoading,
    isLoggedIn,
    createSession,
    onSend,
    // onEnqueue delegates to onSend, which internally routes to the queue
    // endpoint when isInflightRef.current is true.
    onEnqueue: onSend,
    queuedMessages,
    hasMoreMessages: hasMore,
    isLoadingMore,
    loadMore,
    turnStats,
    rateLimitMessage,
    dismissRateLimit,
    // sessionDryRun is the CURRENT session's immutable dry_run flag from API,
    // used to render the banner. The global `isDryRun` preference (for new
    // sessions) lives in the store and is consumed by the toggle button.
    sessionDryRun,
    sessionChatStatus,
  };
}
