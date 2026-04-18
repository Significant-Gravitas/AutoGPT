import {
  getGetV2ListSessionsQueryKey,
  getV2GetPendingMessages,
  postV2QueuePendingMessage,
  useDeleteV2DeleteSession,
  useGetV2ListSessions,
  type getV2ListSessionsResponse,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useQueryClient } from "@tanstack/react-query";
import type { FileUIPart, UIMessage } from "ai";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect, useMemo, useRef, useState } from "react";
import { concatWithAssistantMerge } from "./helpers/convertChatSessionToUiMessages";
import { deduplicateMessages } from "./helpers";
import { useCopilotUIStore } from "./store";
import { useChatSession } from "./useChatSession";
import { useCopilotNotifications } from "./useCopilotNotifications";
import { useCopilotStream } from "./useCopilotStream";
import { useLoadMoreMessages } from "./useLoadMoreMessages";
import { useWorkflowImportAutoSubmit } from "./useWorkflowImportAutoSubmit";

const TITLE_POLL_INTERVAL_MS = 2_000;
const TITLE_POLL_MAX_ATTEMPTS = 5;

interface UploadedFile {
  file_id: string;
  name: string;
  mime_type: string;
}

export function useCopilotPage() {
  const { isUserLoading, isLoggedIn } = useSupabase();
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const [queuedMessages, setQueuedMessages] = useState<string[]>([]);
  const queryClient = useQueryClient();

  const isModeToggleEnabled = useGetFlag(Flag.CHAT_MODE_OPTION);

  const {
    sessionToDelete,
    setSessionToDelete,
    isDrawerOpen,
    setDrawerOpen,
    copilotChatMode,
    copilotLlmModel,
    isDryRun,
  } = useCopilotUIStore();

  const {
    sessionId,
    setSessionId,
    hydratedMessages,
    rawSessionMessages,
    historicalDurations,
    hasActiveStream,
    hasMoreMessages,
    oldestSequence,
    isLoadingSession,
    isSessionError,
    createSession,
    isCreatingSession,
    refetchSession,
    sessionDryRun,
  } = useChatSession({ dryRun: isDryRun });

  const {
    messages: currentMessages,
    setMessages,
    sendMessage,
    stop,
    status,
    error,
    isReconnecting,
    isSyncing,
    isUserStoppingRef,
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

  const { pagedMessages, hasMore, isLoadingMore, loadMore } =
    useLoadMoreMessages({
      sessionId,
      initialOldestSequence: oldestSequence,
      initialHasMore: hasMoreMessages,
      initialPageRawMessages: rawSessionMessages,
    });

  // Ref that mirrors whether a stream turn is currently in-flight.
  // Updated synchronously on every render so it always reflects the latest
  // status — unlike reading `status` inside onSend (which captures the
  // closure's render-cycle value and can be stale for a frame).
  // Setting it to true *before* calling sendMessage prevents rapid
  // double-presses from both routing to /stream before React can re-render
  // with status="submitted".
  const isInflightRef = useRef(false);
  isInflightRef.current = status === "streaming" || status === "submitted";

  // Combine paginated messages with current page messages, merging consecutive
  // assistant UIMessages at the page boundary so reasoning + response parts
  // stay in a single bubble. Paged messages are older history prepended before
  // the current page.
  const rawMessages = concatWithAssistantMerge(pagedMessages, currentMessages);

  // The Claude Agent SDK replays earlier turn content when starting a new
  // turn (via --resume). So Turn N's assistant message accumulates Turn 1..N-1
  // text as a LEADING PREFIX before the actual Turn N response. Strip the
  // replayed leading text from Turn N so each turn stays anchored under its
  // own user bubble and the UI does not show the same greeting twice.
  //
  // We compare by concatenated text content rather than by part index
  // because the replay interleaves step-start / step-finish boundary parts
  // whose ordering differs between the original and replayed rendering,
  // making a part-level strict-prefix match unreliable. Tool call IDs in
  // the parts array differ between a real call and its replay too, but
  // Claude typically only replays TEXT content at the top of the next
  // turn, so text-only comparison catches the common case cleanly.
  const messages = useMemo(() => {
    const deduped = deduplicateMessages(rawMessages);

    const textOf = (msg: UIMessage): string =>
      (msg.parts ?? [])
        .map((p) => ("text" in p && typeof p.text === "string" ? p.text : ""))
        .join("");

    const texts = deduped.map(textOf);

    const result: UIMessage[] = [];
    for (let i = 0; i < deduped.length; i++) {
      const msg = deduped[i];
      if (msg.role !== "assistant") {
        result.push(msg);
        continue;
      }
      const myText = texts[i];
      if (!myText) {
        result.push(msg);
        continue;
      }

      // Find the longest earlier-assistant text that is a leading prefix of
      // this message's concatenated text. Three cases:
      //
      // - earlier == my full text  → pure replay; drop this message.
      // - earlier is a strict prefix of my text  → "replay + new content";
      //   strip the earlier portion so only the new content renders.
      // - my text is a strict prefix of earlier (the streaming replay is
      //   still catching up to the earlier full text)  → drop this
      //   message until it grows past. Without this the user sees the
      //   same greeting twice while the resume-replay streams in.
      let stripLen = 0;
      let fullyReplayed = false;
      for (let j = 0; j < i; j++) {
        if (deduped[j].role !== "assistant") continue;
        const earlierText = texts[j];
        if (!earlierText) continue;

        if (myText.startsWith(earlierText)) {
          if (earlierText.length >= stripLen) {
            if (earlierText.length === myText.length) {
              fullyReplayed = true;
              stripLen = earlierText.length;
            } else {
              stripLen = earlierText.length;
              fullyReplayed = false;
            }
          }
        } else if (earlierText.startsWith(myText)) {
          // Streaming replay is still building up content that matches
          // an earlier message's prefix. Drop to avoid the duplicate
          // flash; once my text exceeds earlier this branch stops
          // firing and the regular prefix-strip takes over.
          fullyReplayed = true;
          break;
        }
      }

      if (fullyReplayed) {
        // Turn N's assistant is an exact replay of an earlier one (no new
        // content yet, or Claude is still mid-stream). Drop so the UI
        // doesn't show the same text twice.
        continue;
      }

      if (stripLen === 0) {
        result.push(msg);
        continue;
      }

      // Strip the leading `stripLen` characters of text from my parts,
      // consuming text parts left-to-right. Non-text parts (step-start,
      // step-finish, tool-*) are preserved unchanged so tool widgets and
      // step boundaries render normally.
      const trimmedParts: UIMessage["parts"] = [];
      let remaining = stripLen;
      for (const part of msg.parts ?? []) {
        if (remaining === 0) {
          trimmedParts.push(part);
          continue;
        }
        if ("text" in part && typeof part.text === "string") {
          if (part.text.length <= remaining) {
            remaining -= part.text.length;
            // Drop this text part entirely.
            continue;
          }
          trimmedParts.push({ ...part, text: part.text.slice(remaining) });
          remaining = 0;
        } else {
          trimmedParts.push(part);
        }
      }

      if (
        trimmedParts.length === 0 ||
        trimmedParts.every(
          (p) =>
            "text" in p && typeof p.text === "string" && p.text.length === 0,
        )
      ) {
        continue;
      }
      result.push({ ...msg, parts: trimmedParts });
    }
    return result;
  }, [rawMessages]);

  useCopilotNotifications(sessionId);

  // --- Delete session ---
  const { mutate: deleteSessionMutation, isPending: isDeleting } =
    useDeleteV2DeleteSession({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          });
          if (sessionToDelete?.id === sessionId) {
            setSessionId(null);
          }
          setSessionToDelete(null);
        },
        onError: (error) => {
          toast({
            title: "Failed to delete chat",
            description:
              error instanceof Error ? error.message : "An error occurred",
            variant: "destructive",
          });
          setSessionToDelete(null);
        },
      },
    });

  // --- Responsive ---
  const breakpoint = useBreakpoint();
  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const pendingFilesRef = useRef<File[]>([]);
  // Pre-built file parts from workflow import (already uploaded, skip re-upload)
  const pendingFilePartsRef = useRef<FileUIPart[]>([]);

  // --- Send pending message after session creation ---
  useEffect(() => {
    if (!sessionId || pendingMessage === null) return;
    const msg = pendingMessage;
    const files = pendingFilesRef.current;
    const prebuiltParts = pendingFilePartsRef.current;
    setPendingMessage(null);
    pendingFilesRef.current = [];
    pendingFilePartsRef.current = [];

    if (prebuiltParts.length > 0) {
      // File already uploaded (e.g. workflow import) — send directly
      sendMessage({ text: msg, files: prebuiltParts });
    } else if (files.length > 0) {
      setIsUploadingFiles(true);
      void uploadFiles(files, sessionId)
        .then((uploaded) => {
          if (uploaded.length === 0) {
            toast({
              title: "File upload failed",
              description: "Could not upload any files. Please try again.",
              variant: "destructive",
            });
            return;
          }
          const fileParts = buildFileParts(uploaded);
          sendMessage({
            text: msg,
            files: fileParts.length > 0 ? fileParts : undefined,
          });
        })
        .finally(() => setIsUploadingFiles(false));
    } else {
      sendMessage({ text: msg });
    }
  }, [sessionId, pendingMessage, sendMessage]);

  // --- Extract prompt from URL hash on mount (e.g. /copilot#prompt=Hello) ---
  useWorkflowImportAutoSubmit({
    createSession,
    setPendingMessage,
    pendingFilePartsRef,
  });

  async function uploadFiles(
    files: File[],
    sid: string,
  ): Promise<UploadedFile[]> {
    const results = await Promise.allSettled(
      files.map(async (file) => {
        try {
          const data = await uploadFileDirect(file, sid);
          if (!data.file_id) throw new Error("No file_id returned");
          return {
            file_id: data.file_id,
            name: data.name || file.name,
            mime_type: data.mime_type || "application/octet-stream",
          } as UploadedFile;
        } catch (err) {
          console.error("File upload failed:", err);
          toast({
            title: "File upload failed",
            description: file.name,
            variant: "destructive",
          });
          throw err;
        }
      }),
    );
    return results
      .filter(
        (r): r is PromiseFulfilledResult<UploadedFile> =>
          r.status === "fulfilled",
      )
      .map((r) => r.value);
  }

  function buildFileParts(uploaded: UploadedFile[]): FileUIPart[] {
    return uploaded.map((f) => ({
      type: "file" as const,
      mediaType: f.mime_type,
      filename: f.name,
      url: `/api/proxy/api/workspace/files/${f.file_id}/download`,
    }));
  }

  async function onSend(message: string, files?: File[]) {
    const trimmed = message.trim();
    if (!trimmed && (!files || files.length === 0)) return;

    // Client-side file limits
    if (files && files.length > 0) {
      const MAX_FILES = 10;
      const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB

      if (files.length > MAX_FILES) {
        toast({
          title: "Too many files",
          description: `You can attach up to ${MAX_FILES} files at once.`,
          variant: "destructive",
        });
        return;
      }

      const oversized = files.filter((f) => f.size > MAX_FILE_SIZE_BYTES);
      if (oversized.length > 0) {
        toast({
          title: "File too large",
          description: `${oversized[0].name} exceeds the 100 MB limit.`,
          variant: "destructive",
        });
        return;
      }
    }

    isUserStoppingRef.current = false;

    if (sessionId) {
      const isInFlight = isInflightRef.current;

      if (isInFlight) {
        // File attachments cannot be included in a queued pending message —
        // the queue API does not support file_ids.  Inform the user and bail.
        if (files && files.length > 0) {
          toast({
            title: "Please wait to attach files",
            description:
              "File attachments can't be queued until the current response finishes.",
            variant: "destructive",
          });
          return;
        }

        // Queue the message into the pending buffer so it is picked up between
        // tool-call rounds by the currently running executor turn.
        try {
          await postV2QueuePendingMessage(sessionId, { message: trimmed });
          setQueuedMessages((prev) => [...prev, trimmed]);
        } catch (err) {
          toast({
            title: "Could not queue message",
            description: "Please wait for the current response to finish.",
            variant: "destructive",
          });
          throw err;
        }
        return;
      }

      // Mark in-flight synchronously before sendMessage so any rapid
      // second press sees isInflightRef.current=true and routes to /pending
      // instead of triggering a duplicate /stream POST.
      isInflightRef.current = true;
      if (files && files.length > 0) {
        setIsUploadingFiles(true);
        try {
          const uploaded = await uploadFiles(files, sessionId);
          if (uploaded.length === 0) {
            // All uploads failed — abort send so chips revert to editable
            isInflightRef.current = false;
            throw new Error("All file uploads failed");
          }
          const fileParts = buildFileParts(uploaded);
          sendMessage({
            text: trimmed || "",
            files: fileParts.length > 0 ? fileParts : undefined,
          });
        } finally {
          setIsUploadingFiles(false);
        }
      } else {
        sendMessage({ text: trimmed });
      }
      return;
    }

    setPendingMessage(trimmed || "");
    if (files && files.length > 0) {
      pendingFilesRef.current = files;
    }
    await createSession();
  }

  // --- Session list (for mobile drawer & sidebar) ---
  const { data: sessionsResponse, isLoading: isLoadingSessions } =
    useGetV2ListSessions(
      { limit: 50 },
      { query: { enabled: !isUserLoading && isLoggedIn } },
    );

  const sessions =
    sessionsResponse?.status === 200 ? sessionsResponse.data.sessions : [];

  // Sync the queued-messages indicator with the backend buffer on session
  // load/change and whenever a turn ends (the backend drains at turn start
  // and auto-continue, so the buffer may shrink without frontend action).
  const prevQueuePeekSessionIdRef = useRef<string | null>(sessionId);
  const hasSeenTurnStartAssistantRef = useRef(false);
  useEffect(() => {
    const sessionChanged = prevQueuePeekSessionIdRef.current !== sessionId;
    prevQueuePeekSessionIdRef.current = sessionId;

    if (sessionChanged) {
      // Reset per-stream promotion state and clear any chips rendered for
      // the previous session before the peek resolves — otherwise the new
      // session briefly shows stale chips against its own messages.
      hasSeenTurnStartAssistantRef.current = false;
      setQueuedMessages([]);
    }

    if (!sessionId) return;
    const isIdle = status === "ready" || status === "error";
    if (!sessionChanged && !isIdle) return;

    void getV2GetPendingMessages(sessionId).then((res) => {
      setQueuedMessages(
        res.status === 200 && res.data.count > 0 ? res.data.messages : [],
      );
    });
  }, [sessionId, status]);

  // Promote queued chips to real user bubbles when the backend consumes them.
  //
  // The backend now combines ALL pending messages into a single auto-continue
  // turn (joined by "\n\n"), so we promote all chips at once into one user
  // bubble that matches the combined message the backend persists to the DB.
  //
  // Detection:
  // - submitted → streaming: turn-start drain merged chips into the CURRENT
  //   user message. Clear chips once the buffer is empty.
  // - Auto-continue is detected by observing a SECOND new assistant ID after
  //   the first one for this chain of turns. The first-new is Turn 1's
  //   assistant (created when status goes submitted→streaming). Any later
  //   new assistant ID while still streaming must be the auto-continue turn.
  const prevStatusForQueueRef = useRef(status);
  const seenAssistantIdsRef = useRef<Set<string>>(
    new Set(messages.filter((m) => m.role === "assistant").map((m) => m.id)),
  );
  const queuedMessagesRef = useRef(queuedMessages);
  queuedMessagesRef.current = queuedMessages;

  useEffect(() => {
    const prevStatus = prevStatusForQueueRef.current;
    prevStatusForQueueRef.current = status;

    const currentAssistantIds = messages
      .filter((m) => m.role === "assistant")
      .map((m) => m.id);
    const newAssistantIds = currentAssistantIds.filter(
      (id) => !seenAssistantIdsRef.current.has(id),
    );
    currentAssistantIds.forEach((id) => seenAssistantIdsRef.current.add(id));

    if (!sessionId) return;

    const isTurnStarting = prevStatus === "submitted" && status === "streaming";
    const isActive = status === "streaming" || status === "submitted";
    const becameIdle =
      (prevStatus === "streaming" || prevStatus === "submitted") &&
      (status === "ready" || status === "error");

    if (isTurnStarting) {
      // Reset the "seen first assistant for this turn" marker so we can
      // distinguish Turn 1's assistant from the auto-continue one.
      hasSeenTurnStartAssistantRef.current = false;
      // Turn-start drain: chips were merged into the submitted message.
      // Peek the buffer — clear chips only if the backend drained them.
      void getV2GetPendingMessages(sessionId).then((res) => {
        if (res.status === 200 && res.data.count === 0) {
          setQueuedMessages([]);
        }
      });
    }

    if (becameIdle) {
      // Stream ended — hydration will replace messages with DB state.
      hasSeenTurnStartAssistantRef.current = false;
    }

    if (!isActive || newAssistantIds.length === 0) return;

    // The first new assistant of a stream chain is Turn 1's opener — not an
    // auto-continue. Remember it, then wait for the next new assistant.
    let candidateIds = newAssistantIds;
    if (!hasSeenTurnStartAssistantRef.current) {
      hasSeenTurnStartAssistantRef.current = true;
      // If Turn 1 plus its auto-continue assistant land in the same render
      // batch, drop the first and treat the rest as auto-continue. Otherwise
      // wait for the next render.
      if (candidateIds.length === 1) return;
      candidateIds = candidateIds.slice(1);
    }

    if (queuedMessagesRef.current.length === 0) return;

    // Auto-continue detected: promote ALL chips as one combined user bubble
    // matching the backend's "\n\n".join(texts). Insert immediately before
    // the auto-continue assistant ID so it sits between Turn N and Turn N+1
    // visually.
    const autoContinueId = candidateIds[0];
    const combinedText = queuedMessagesRef.current.join("\n\n");

    setMessages((prev) => {
      if (prev.some((m) => m.id === `promoted-${autoContinueId}`)) return prev;
      const result = [...prev];
      const idx = result.findIndex((m) => m.id === autoContinueId);
      const insertAt = idx === -1 ? result.length : idx;
      result.splice(insertAt, 0, {
        id: `promoted-${autoContinueId}`,
        role: "user" as const,
        parts: [
          {
            type: "text" as const,
            text: combinedText,
            state: "done" as const,
          },
        ],
      });
      return result;
    });
    setQueuedMessages([]);
  }, [messages, status, sessionId, setMessages]);

  // Mid-turn drain detection: SDK tool wrapper drains the pending buffer at
  // a tool boundary and the service-layer persist adds the follow-up as a
  // real user row in session.messages — but this does NOT emit an SSE event,
  // so the AI SDK's live messages list never hears about it and the chip
  // keeps rendering on the client until a hard refresh hydrates from DB.
  //
  // Fix: while the stream is active AND we still have chips locally, poll
  // the backend buffer. If it drops to zero we know the backend drained
  // (either via the MCP wrapper on a tool boundary, or via turn-end
  // auto-continue) — promote chips locally as a user bubble so the UI
  // matches the persisted DB state in real time. Force-hydrate after stream
  // end will later replace the promoted bubble with the real DB row (same
  // content, no flicker).
  useEffect(() => {
    if (!sessionId) return;
    if (status !== "streaming" && status !== "submitted") return;
    if (queuedMessages.length === 0) return;

    const interval = setInterval(async () => {
      try {
        const res = await getV2GetPendingMessages(sessionId);
        if (res.status !== 200 || res.data.count !== 0) return;
        if (queuedMessagesRef.current.length === 0) return;
        const combined = queuedMessagesRef.current.join("\n\n");
        setMessages((prev) => {
          // Don't double-promote: the turn-end auto-continue promotion uses
          // `promoted-${assistantId}` ids, so any id prefix match is enough.
          if (prev.some((m) => m.id.startsWith("promoted-"))) return prev;
          return [
            ...prev,
            {
              id: `promoted-midturn-${Date.now()}`,
              role: "user" as const,
              parts: [
                {
                  type: "text" as const,
                  text: combined,
                  state: "done" as const,
                },
              ],
            },
          ];
        });
        setQueuedMessages([]);
      } catch {
        // Poll failures are harmless — we try again on the next tick or
        // fall back to hydration-on-stream-end.
      }
    }, 2_000);

    return () => clearInterval(interval);
  }, [sessionId, status, queuedMessages.length, setMessages]);

  // Start title polling when stream ends cleanly — sidebar title animates in
  const titlePollRef = useRef<ReturnType<typeof setInterval>>();
  const prevStatusRef = useRef(status);

  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;

    const wasActive = prev === "streaming" || prev === "submitted";
    const isNowReady = status === "ready";

    if (!wasActive || !isNowReady || !sessionId || isReconnecting) return;

    queryClient.invalidateQueries({
      queryKey: getGetV2ListSessionsQueryKey({ limit: 50 }),
    });
    const sid = sessionId;
    let attempts = 0;
    clearInterval(titlePollRef.current);
    titlePollRef.current = setInterval(() => {
      const data = queryClient.getQueryData<getV2ListSessionsResponse>(
        getGetV2ListSessionsQueryKey({ limit: 50 }),
      );
      const hasTitle =
        data?.status === 200 &&
        data.data.sessions.some((s) => s.id === sid && s.title);
      if (hasTitle || attempts >= TITLE_POLL_MAX_ATTEMPTS) {
        clearInterval(titlePollRef.current);
        titlePollRef.current = undefined;
        return;
      }
      attempts += 1;
      queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey({ limit: 50 }),
      });
    }, TITLE_POLL_INTERVAL_MS);
  }, [status, sessionId, isReconnecting, queryClient]);

  // Clean up polling on session change or unmount
  useEffect(() => {
    return () => {
      clearInterval(titlePollRef.current);
      titlePollRef.current = undefined;
    };
  }, [sessionId]);

  // --- Mobile drawer handlers ---
  function handleOpenDrawer() {
    setDrawerOpen(true);
  }

  function handleCloseDrawer() {
    setDrawerOpen(false);
  }

  function handleDrawerOpenChange(open: boolean) {
    setDrawerOpen(open);
  }

  function handleSelectSession(id: string) {
    setSessionId(id);
    if (isMobile) setDrawerOpen(false);
  }

  function handleNewChat() {
    setSessionId(null);
    if (isMobile) setDrawerOpen(false);
  }

  // --- Delete handlers ---
  function handleDeleteClick(id: string, title: string | null | undefined) {
    if (isDeleting) return;
    setSessionToDelete({ id, title });
  }

  function handleConfirmDelete() {
    if (sessionToDelete) {
      deleteSessionMutation({ sessionId: sessionToDelete.id });
    }
  }

  function handleCancelDelete() {
    if (!isDeleting) {
      setSessionToDelete(null);
    }
  }

  return {
    sessionId,
    messages,
    status,
    error,
    stop,
    isReconnecting,
    isSyncing,
    isLoadingSession,
    isSessionError,
    isCreatingSession,
    isUploadingFiles,
    isUserLoading,
    isLoggedIn,
    createSession,
    onSend,
    // onEnqueue delegates to onSend, which internally routes to the pending
    // endpoint when isInflightRef.current is true.
    onEnqueue: onSend,
    queuedMessages,
    // Pagination
    hasMoreMessages: hasMore,
    isLoadingMore,
    loadMore,
    // Mobile drawer
    isMobile,
    isDrawerOpen,
    sessions,
    isLoadingSessions,
    handleOpenDrawer,
    handleCloseDrawer,
    handleDrawerOpenChange,
    handleSelectSession,
    handleNewChat,
    // Delete functionality
    sessionToDelete,
    isDeleting,
    handleDeleteClick,
    handleConfirmDelete,
    handleCancelDelete,
    // Historical durations for persisted timer stats
    historicalDurations,
    // Rate limit reset
    rateLimitMessage,
    dismissRateLimit,
    // Dry run dev toggle
    // isDryRun = global preference for NEW sessions (from localStorage).
    // sessionDryRun = actual dry_run value of the CURRENT session (from API).
    // Use isDryRun to configure future sessions; use sessionDryRun to display
    // the current session's simulation state (banner, indicators).
    isDryRun,
    sessionDryRun,
  };
}
