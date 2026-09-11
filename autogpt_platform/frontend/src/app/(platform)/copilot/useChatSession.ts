import {
  getGetV2GetSessionQueryKey,
  useGetV2ListChatTransports,
  useGetV2GetSession,
  useGetV2ListSessions,
  usePostV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { CreateSessionRequest } from "@/app/api/__generated__/models/createSessionRequest";
import type { CreateSessionRequestExecutionTarget } from "@/app/api/__generated__/models/createSessionRequestExecutionTarget";
import type { PublicChatSessionMetadata } from "@/app/api/__generated__/models/publicChatSessionMetadata";
import { SESSION_LIST_QUERY_KEY } from "./useSessionList";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useMemo, useRef } from "react";
import {
  convertChatSessionMessagesToUiMessages,
  type TurnStatsMap,
} from "./helpers/convertChatSessionToUiMessages";
import { resolveSessionDryRun } from "./helpers";
import { useCopilotUIStore, type NewChatExecutionTarget } from "./store";
import {
  getAvailableLLMTransports,
  resolveCopilotLLMAuthSelection,
} from "./helpers/copilotLlmAuth";
import { useCopilotStreamStore } from "./copilotStreamStore";
import { latestExpertSessionParams } from "./expertSessionQuery";

interface UseChatSessionOptions {
  dryRun?: boolean;
  executionTarget?: NewChatExecutionTarget;
  expertId?: string | null;
  /** Off = keep the fresh new-task state addressed to the expert instead of
   *  jumping into their latest thread (``/copilot?expertId=…&new=1``). */
  adoptLatestExpertThread?: boolean;
}

class ExecutionTargetSelectionError extends Error {}

export function useChatSession({
  dryRun = false,
  executionTarget,
  expertId = null,
  adoptLatestExpertThread = true,
}: UseChatSessionOptions = {}) {
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const queryClient = useQueryClient();
  const setExecutionTargetPickerOpen = useCopilotUIStore(
    (state) => state.setExecutionTargetPickerOpen,
  );
  const setExecutionTargetError = useCopilotUIStore(
    (state) => state.setExecutionTargetError,
  );
  const setNewChatExecutionTarget = useCopilotUIStore(
    (state) => state.setNewChatExecutionTarget,
  );
  const copilotLlmAuth = useCopilotUIStore((state) => state.copilotLlmAuth);

  const transportQuery = useGetV2ListChatTransports({
    query: {
      enabled: !sessionId,
      refetchOnWindowFocus: true,
      staleTime: 0,
    },
  });
  const chatTransports =
    transportQuery.data?.status === 200
      ? transportQuery.data.data.transports
      : undefined;

  const sessionQuery = useGetV2GetSession(sessionId ?? "", undefined, {
    query: {
      enabled: !!sessionId,
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      refetchOnMount: true,
    },
  });

  // When dry-run mode is toggled, discard the current session so the next
  // send creates a fresh one with the correct dry_run flag.  Sessions are
  // immutable once created: dry_run cannot be changed after the fact.
  const prevDryRunRef = useRef(dryRun);
  useEffect(() => {
    if (prevDryRunRef.current !== dryRun) {
      prevDryRunRef.current = dryRun;
      if (sessionId) {
        setSessionId(null);
      }
    }
  }, [dryRun, sessionId, setSessionId]);

  // Invalidate query cache on session switch.
  // useChat destroys its Chat instance on id change, so messages are lost.
  // We invalidate BOTH the old session (stale after leaving) and the new
  // session (may have been cached before the user sent their last message,
  // so active_stream and messages could be outdated). This guarantees a
  // fresh fetch that gives the resume effect accurate hasActiveStream state.
  const prevSessionIdRef = useRef(sessionId);

  useEffect(() => {
    const prev = prevSessionIdRef.current;
    prevSessionIdRef.current = sessionId;
    if (prev && prev !== sessionId) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(prev),
      });
    }
    if (sessionId) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
    }
  }, [sessionId, queryClient]);

  // Deep link /copilot?expertId=<id>: adopt the expert's latest thread. Only
  // the mount-time expertId adopts — a recipient picked in the UI after mount
  // must keep the fresh new-task state, not jump to the expert's old thread.
  // The once-per-expert latch lives in the UI store, NOT in a ref: the chat
  // host remounts on every sessionId change (CopilotPage keys the subtree),
  // and "New Chat" clears the expertId param asynchronously via nuqs — a
  // ref-based latch is wiped by the remount before the param clears, which
  // bounced New Chat straight back into the adopted thread.
  const mountExpertIdRef = useRef(expertId);
  const adoptedExpertThreads = useCopilotUIStore((s) => s.adoptedExpertThreads);
  const markExpertThreadAdopted = useCopilotUIStore(
    (s) => s.markExpertThreadAdopted,
  );
  // Once the user commits to a new thread by hitting send, adoption must stop:
  // `useSendMessage` flushes its pending first message on *any* sessionId
  // change, so a late adoption would post that message into the old thread.
  const sendStartedRef = useRef(false);
  const canAdoptExpertSession =
    adoptLatestExpertThread &&
    !!expertId &&
    !sessionId &&
    expertId === mountExpertIdRef.current &&
    !adoptedExpertThreads.has(expertId);

  const latestExpertSessionQuery = useGetV2ListSessions(
    latestExpertSessionParams(expertId),
    {
      query: {
        enabled: canAdoptExpertSession,
        refetchOnWindowFocus: false,
      },
    },
  );

  useEffect(() => {
    if (!canAdoptExpertSession || !expertId) return;
    if (sendStartedRef.current) return;
    if (latestExpertSessionQuery.data?.status !== 200) return;
    const latest = latestExpertSessionQuery.data.data.sessions[0];
    if (!latest) return;
    markExpertThreadAdopted(expertId);
    setSessionId(latest.id);
  }, [
    canAdoptExpertSession,
    expertId,
    latestExpertSessionQuery.data,
    markExpertThreadAdopted,
    setSessionId,
  ]);

  // True while the deep-link adoption could still navigate away from the
  // new-task screen. Callers disable the composer for that window so a draft
  // (or a send) can't be swallowed by the navigation. Keyed on `isLoading`
  // rather than `isPending` so a failing request releases the composer instead
  // of locking it for the length of the retry schedule; `sendStartedRef` is
  // what actually guarantees a send is never misrouted.
  const isAdoptingExpertSession =
    canAdoptExpertSession &&
    !sendStartedRef.current &&
    latestExpertSessionQuery.isLoading;

  const freshSessionData =
    !!sessionId && sessionQuery.data?.status === 200 && !sessionQuery.isFetching
      ? sessionQuery.data.data
      : null;

  // Expose active_stream info so the caller can trigger manual resume
  // after hydration completes (rather than relying on AI SDK's built-in
  // resume which fires before hydration).
  const hasActiveStream = useMemo(() => {
    return !!freshSessionData?.active_stream;
  }, [freshSessionData]);

  // Backend-reported start time of the active turn. Used to seed the
  // elapsed-time counter on mount so restored sessions show honest
  // "time since the backend started the turn" rather than "time since
  // this mount subscribed to the SSE".
  const activeStreamStartedAt = useMemo(() => {
    return freshSessionData?.active_stream?.started_at ?? null;
  }, [freshSessionData]);

  // Pagination metadata from the initial page load
  const hasMoreMessages = useMemo(() => {
    return !!freshSessionData?.has_more_messages;
  }, [freshSessionData]);

  const oldestSequence = useMemo(() => {
    return freshSessionData?.oldest_sequence ?? null;
  }, [freshSessionData]);

  // Memoize so the effect in useCopilotPage doesn't infinite-loop on a new
  // array reference every render. Re-derives only when query data changes.
  // When the session is complete (no active stream), mark dangling tool
  // calls as completed so stale spinners don't persist after refresh.
  const { hydratedMessages, historicalTurnStats, activeTurnStartMessageId } =
    useMemo(() => {
      if (!freshSessionData || !sessionId)
        return {
          hydratedMessages: undefined,
          historicalTurnStats: new Map() as TurnStatsMap,
          activeTurnStartMessageId: null,
        };
      const result = convertChatSessionMessagesToUiMessages(
        sessionId,
        freshSessionData.messages ?? [],
        {
          isComplete: !hasActiveStream,
          activeTurnStartedAt: activeStreamStartedAt,
        },
      );
      return {
        hydratedMessages: result.messages,
        historicalTurnStats: result.stats,
        activeTurnStartMessageId: result.activeTurnStartId,
      };
    }, [freshSessionData, sessionId, hasActiveStream, activeStreamStartedAt]);

  const { mutateAsync: createSessionMutation, isPending: isCreatingSession } =
    usePostV2CreateSession();

  async function createSession(options?: { expertKickoff?: boolean }) {
    if (sessionId) return sessionId;
    // Latched for the life of this mount, including on failure: once the user
    // has asked for a new thread, auto-navigating them into an old one is
    // never the right recovery.
    sendStartedRef.current = true;
    const resolvedLLMAuth = resolveCopilotLLMAuthSelection(
      chatTransports,
      copilotLlmAuth,
    );
    const availableTransports = getAvailableLLMTransports(chatTransports);
    if (transportQuery.isError && chatTransports === undefined) {
      toast({
        variant: "destructive",
        title: "Could not check AI connections",
        description: "Refresh the page and try again.",
      });
      throw new Error("Could not load AI connections");
    }
    if (chatTransports !== undefined && availableTransports.length === 0) {
      toast({
        variant: "destructive",
        title: "Otto needs an AI connection",
        description:
          "Sign in with ChatGPT under OpenAI in Settings → Integrations, or configure a chat API or local model on this server.",
      });
      throw new Error("chat_transport_not_configured");
    }
    if (!resolvedLLMAuth) {
      const connectionsAreLoading = chatTransports === undefined;
      toast({
        variant: "destructive",
        title: connectionsAreLoading
          ? "AI connections are still loading"
          : "Choose an AI connection",
        description: connectionsAreLoading
          ? "Wait a moment and try again."
          : "Select the connection Otto should use before starting a new task.",
      });
      throw new Error(
        connectionsAreLoading
          ? "AI connections are still loading"
          : "AI connection selection required",
      );
    }
    if (
      copilotLlmAuth !== null &&
      copilotLlmAuth.authProvider !== "platform" &&
      resolvedLLMAuth.authProvider === "platform"
    ) {
      toast({
        title: "AI connections changed",
        description:
          "The next Otto task will resolve the currently available connection before it starts.",
      });
    }

    try {
      const target = sessionExecutionTargetRequest(executionTarget);
      if (executionTarget?.kind === "local" && !target) {
        const message =
          "Choose a connected computer and folder before starting a Local PC chat.";
        setExecutionTargetError(message);
        setExecutionTargetPickerOpen(true);
        toast({
          variant: "destructive",
          title: "Choose a Local PC folder",
          description: message,
        });
        throw new ExecutionTargetSelectionError(message);
      }

      const sessionData: CreateSessionRequest = {};
      // Only an explicit choice travels. Naming the route unconditionally
      // makes every new chat an override, which is how a connection picked
      // once quietly became the account's default and how a default changed
      // in Settings stopped taking effect: the server skips its own default
      // whenever the client names one. `copilotLlmAuth` is null until the
      // user actually picks, and null means "use whatever the server says".
      if (copilotLlmAuth !== null) {
        sessionData.llm_auth_provider = resolvedLLMAuth.authProvider;
        if (resolvedLLMAuth.authProvider === "codex") {
          sessionData.llm_credential_id = resolvedLLMAuth.credentialId;
        }
      }
      if (target) sessionData.execution_target = target;
      if (dryRun) sessionData.dry_run = true;
      if (expertId) sessionData.expert_id = expertId;
      if (options?.expertKickoff) sessionData.expert_kickoff = true;
      const body =
        Object.keys(sessionData).length > 0
          ? { data: sessionData }
          : { data: null };
      const response = await createSessionMutation(body);
      if (response.status !== 200 || !response.data?.id) {
        const error = new Error("Failed to create session");
        Sentry.captureException(error, {
          extra: { status: response.status },
        });
        toast({
          variant: "destructive",
          title: "Could not start a new chat session",
          description: "Please try again.",
        });
        throw error;
      }
      useCopilotStreamStore
        .getState()
        .bindPendingFirstSendToSession(response.data.id);
      setSessionId(response.data.id);
      queryClient.invalidateQueries({
        queryKey: SESSION_LIST_QUERY_KEY,
      });
      return response.data.id;
    } catch (error) {
      if (error instanceof ExecutionTargetSelectionError) throw error;
      if (
        error instanceof ApiError &&
        error.status === 409 &&
        executionTarget?.kind === "local"
      ) {
        const message =
          "The selected computer or folder changed. Reconnect it and choose the folder again.";
        setNewChatExecutionTarget({
          ...executionTarget,
          connectionID: null,
          browseID: null,
          directoryRef: null,
          displayPath: null,
        });
        setExecutionTargetError(message);
        setExecutionTargetPickerOpen(true);
        toast({
          variant: "destructive",
          title: "Local PC connection changed",
          description: message,
        });
        throw error;
      }
      if (
        error instanceof Error &&
        error.message === "Failed to create session"
      ) {
        throw error; // already handled above
      }
      Sentry.captureException(error);
      toast({
        variant: "destructive",
        title: "Could not start a new chat session",
        description: "Please try again.",
      });
      throw error;
    }
  }

  // Raw messages from the initial page — exposed for cross-page
  // tool output matching by useLoadMoreMessages.
  const rawSessionMessages =
    freshSessionData?.messages != null
      ? ((freshSessionData.messages ?? []) as unknown[])
      : [];

  // The actual dry_run value stored in the session's metadata, read directly
  // from the API response. This reflects what the session was ACTUALLY created
  // with — not the user's current UI preference (isDryRun store).
  //
  // Design intent: the global isDryRun store is only used when creating NEW
  // sessions. Once a session exists, its dry_run flag is immutable and should
  // be read from here rather than from the store, which may have changed.
  const sessionDryRun = useMemo(
    () => (freshSessionData ? resolveSessionDryRun(sessionQuery.data) : false),
    [sessionQuery.data, freshSessionData],
  );

  const sessionExecutionTarget =
    sessionId && sessionQuery.data?.status === 200
      ? readSessionExecutionTarget(sessionQuery.data.data.metadata)
      : null;

  const sessionChatStatus = (
    freshSessionData as { chat_status?: string } | undefined
  )?.chat_status;

  const sessionLlmAuthProvider: "platform" | "codex" | null =
    sessionId && sessionQuery.data?.status === 200
      ? sessionQuery.data.data.metadata?.llm_auth_provider === "codex"
        ? "codex"
        : "platform"
      : null;
  const sessionLlmCredentialId =
    sessionId && sessionQuery.data?.status === 200
      ? (sessionQuery.data.data.metadata?.llm_credential_id ?? null)
      : null;

  // The expert this session actually belongs to, straight off the session
  // response rather than the URL — the ?expertId= param only describes what
  // the NEXT session will be and is absent on most ways of reaching a thread.
  // Read from the query rather than `freshSessionData` (which nulls out during
  // background refetches) so the expert header doesn't blink.
  const sessionExpertId =
    sessionQuery.data?.status === 200
      ? (sessionQuery.data.data.expert_id ?? null)
      : null;

  return {
    sessionId,
    setSessionId,
    sessionLlmAuthProvider,
    sessionLlmCredentialId,
    sessionExpertId,
    isAdoptingExpertSession,
    hydratedMessages,
    rawSessionMessages,
    historicalTurnStats,
    activeTurnStartMessageId,
    hasActiveStream,
    activeStreamStartedAt,
    hasMoreMessages,
    oldestSequence,
    // Only treat the session as loading during the INITIAL fetch (no cached
    // data yet). Background refetches keep the input enabled — otherwise a
    // fill+Enter race can trigger handleSend while ``disabled`` briefly
    // flips back to ``true`` mid-refetch, silently dropping the message.
    isLoadingSession: sessionQuery.isLoading,
    isSessionError: sessionQuery.isError,
    createSession,
    isCreatingSession,
    refetchSession: sessionQuery.refetch,
    sessionDryRun,
    sessionExecutionTarget,
    sessionChatStatus,
  };
}

function sessionExecutionTargetRequest(
  target: NewChatExecutionTarget | undefined,
): CreateSessionRequestExecutionTarget | null {
  if (!target) return null;
  if (target.kind === "cloud") return { kind: "cloud" };
  if (
    !target.machineID ||
    !target.connectionID ||
    !target.browseID ||
    !target.directoryRef
  ) {
    return null;
  }
  return {
    kind: "local",
    machine_id: target.machineID,
    expected_connection_id: target.connectionID,
    browse_id: target.browseID,
    directory_ref: target.directoryRef,
  };
}

function readSessionExecutionTarget(metadata?: PublicChatSessionMetadata) {
  const target = metadata?.execution_target;
  if (!target || target.kind === "cloud") return { kind: "cloud" as const };
  if (!("machine_id" in target)) return null;
  return {
    kind: "local" as const,
    machine_id: target.machine_id,
    allowed_root: target.allowed_root,
  };
}
