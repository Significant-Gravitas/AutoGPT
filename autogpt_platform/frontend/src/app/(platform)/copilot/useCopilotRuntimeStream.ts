import { getGetV2GetCopilotUsageQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import type { UIMessage } from "ai";
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

import { handleStreamError } from "./copilotStreamErrorHandlers";
import { useCopilotStreamStore } from "./copilotStreamStore";
import {
  clearKickoffPending,
  getKickoffAttemptTokenFromMetadata,
  getKickoffExpertId,
  getKickoffExpertIdFromMetadata,
  withKickoffLock,
} from "./expertKickoff";
import {
  convertChatSessionMessagesToUiMessages,
  type TurnStatsMap,
} from "./helpers/convertChatSessionToUiMessages";
import {
  latestProviderFailure,
  parseProviderFailure,
  providerFailureFingerprint,
  type ProviderFailure,
} from "./providerFailure";
import { useCopilotUIStore, type CopilotLlmModel } from "./store";
import {
  createAssistantRunJoiner,
  createTailRenderer,
} from "./stream/renderTail";
import {
  getTurnRuntime,
  type RuntimePhase,
  type RuntimeSnapshot,
  type SendInput,
  type SessionView,
  type TurnRuntime,
} from "./stream/turnRuntime";
import type { PersistedRow } from "./stream/turnLog";
import { useCopilotStop } from "./useCopilotStop";

const NOTICE_TEXT = {
  "catching-up": "Catching up…",
  reconnecting: "Reconnecting…",
  offline: "Offline — waiting for the network…",
} as const;

const EMPTY_SNAPSHOT: RuntimeSnapshot = {
  segments: [],
  ownedFrom: null,
  phase: "idle",
  notice: null,
  error: null,
  stopped: false,
};

interface Args {
  userId?: string | null;
  sessionId: string | null;
  /** The fresh session GET: its rows are the history above the runtime's tail. */
  sessionView: SessionView | null;
  hydratedMessages: UIMessage[] | undefined;
  rawSessionMessages?: unknown[];
  sessionAuthProvider?: string | null;
  sessionCredentialId?: string | null;
  refetchSession: () => Promise<{ data?: unknown }>;
  copilotModel: CopilotLlmModel | undefined;
}

/**
 * The chat stream on our own runtime: one connection per session, resumed
 * at a cursor, rendered from the rows the backend persists. Returns the same
 * shape as `useCopilotStream` so the page can take either.
 */
export function useCopilotRuntimeStream({
  userId = null,
  sessionId,
  sessionView,
  hydratedMessages,
  rawSessionMessages,
  sessionAuthProvider = null,
  sessionCredentialId = null,
  refetchSession,
  copilotModel,
}: Args) {
  const queryClient = useQueryClient();
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);
  const [rateLimitMessage, setRateLimitMessage] = useState<string | null>(null);
  const [platformLimitFailure, setPlatformLimitFailure] =
    useState<ProviderFailure | null>(null);
  const [providerLimit, setProviderLimit] = useState<ProviderFailure | null>(
    null,
  );
  const dismissedProviderFailureRef = useRef<{
    sessionId: string;
    fingerprint: string;
  } | null>(null);

  const runtime = useMemo(
    () => (sessionId ? getTurnRuntime(sessionId) : null),
    [sessionId],
  );
  const snapshot = useSyncExternalStore(
    runtime?.subscribe ?? subscribeNothing,
    runtime?.getSnapshot ?? emptySnapshot,
    runtime?.getSnapshot ?? emptySnapshot,
  );
  const isUserStoppingRef = useRef(false);
  isUserStoppingRef.current = snapshot.stopped;

  useEffect(() => {
    if (!runtime) return;
    return runtime.bind(
      {
        onError: handleError,
        onTurnEnd: () =>
          queryClient.invalidateQueries({
            queryKey: getGetV2GetCopilotUsageQueryKey(),
          }),
      },
      async () => viewOf((await refetchSession()).data),
    );
  });

  // Before paint, so a running turn's seeded rows replace its history rows
  // in the same frame.
  useLayoutEffect(() => {
    if (runtime && sessionView) runtime.observe(sessionView);
  }, [runtime, sessionView]);

  const cut = snapshot.ownedFrom ?? prospectiveCut(runtime, sessionView);
  const history = useSessionHistory(sessionId, sessionView, cut);
  const renderTail = useMemo(
    () => createTailRenderer(sessionId ?? ""),
    [sessionId],
  );
  const tail = useMemo(() => renderTail(snapshot), [renderTail, snapshot]);
  const joinAssistantRuns = useMemo(() => createAssistantRunJoiner(), []);
  const messages = useMemo(
    () =>
      sessionId
        ? joinAssistantRuns([...history.messages, ...tail.messages])
        : [],
    [sessionId, joinAssistantRuns, history.messages, tail.messages],
  );
  const turnStats = useMemo(
    () => withLatestUser(messages, history.stats, tail.stats),
    [messages, history.stats, tail.stats],
  );

  function handleError(error: Error, rawFailure: unknown) {
    if (!sessionId) return;
    const coord = useCopilotStreamStore.getState().getCoord(sessionId);
    const kickoffExpertId = coord.lastSubmittedKickoffExpertId;
    const kickoffAttemptToken = coord.lastSubmittedKickoffAttemptToken;
    const clearRecovery = () =>
      useCopilotStreamStore.getState().updateCoord(sessionId, {
        lastSubmittedMessageText: null,
        lastSubmittedKickoffExpertId: null,
        lastSubmittedKickoffAttemptToken: null,
      });
    handleStreamError({
      error,
      providerFailure: parseProviderFailure(rawFailure),
      onRateLimit: (message, limitFailure, origin) => {
        // The backend refuses a turn over our cap before persisting its
        // prompt: put the text back in an empty composer and drop the bubble.
        if (coord.lastSubmittedMessageText) {
          if (kickoffExpertId) clearRecovery();
          else if (isComposerEmpty()) {
            setInitialPrompt(coord.lastSubmittedMessageText);
            clearRecovery();
          }
          runtime?.dropUnsentUserRow();
        }
        if (limitFailure && origin === "provider") {
          dismissedProviderFailureRef.current = null;
          setProviderLimit(limitFailure);
        } else {
          setPlatformLimitFailure(limitFailure ?? null);
          setRateLimitMessage(limitFailure?.message || message);
        }
      },
      onReconnect: () => runtime?.ensureConnected("error"),
      isUserStoppingRef,
    });
    if (!kickoffExpertId) return;
    if (userId && kickoffAttemptToken) {
      void withKickoffLock(userId, kickoffExpertId, async () => {
        clearKickoffPending(userId, kickoffExpertId, kickoffAttemptToken);
      }).catch(() => undefined);
    }
    clearRecovery();
    runtime?.dropUnsentUserRow();
  }

  async function sendMessage(input: SendInput) {
    if (!runtime || !sessionId) return;
    const metadata = input.metadata;
    useCopilotStreamStore.getState().updateCoord(sessionId, {
      lastSubmittedMessageText: "text" in input ? input.text : textOf(input),
      lastSubmittedKickoffExpertId: getKickoffExpertIdFromMetadata(metadata),
      lastSubmittedKickoffAttemptToken:
        getKickoffAttemptTokenFromMetadata(metadata),
    });
    await runtime.send(input, copilotModel);
  }

  const stop = useCopilotStop({
    sessionId,
    stopStream: () => runtime?.stop(),
    isUserStoppingRef,
    setIsUserStopping: () => {},
  });

  // A failure the chat is still sitting on survives a reload: the backend
  // persisted its envelope onto the marker row.
  useEffect(() => {
    setProviderLimit(null);
    dismissedProviderFailureRef.current = null;
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId || !rawSessionMessages?.length) return;
    const historical = latestProviderFailure(
      rawSessionMessages,
      sessionAuthProvider
        ? {
            authProvider: sessionAuthProvider,
            credentialId: sessionCredentialId,
          }
        : null,
    );
    const dismissed = dismissedProviderFailureRef.current;
    if (
      historical &&
      dismissed?.sessionId === sessionId &&
      dismissed.fingerprint === providerFailureFingerprint(historical)
    ) {
      return;
    }
    setProviderLimit((current) => current ?? historical);
  }, [rawSessionMessages, sessionAuthProvider, sessionCredentialId, sessionId]);

  useEffect(() => {
    if (!sessionId || !hydratedMessages) return;
    const kickoffExpertId = useCopilotStreamStore
      .getState()
      .getCoord(sessionId).lastSubmittedKickoffExpertId;
    if (!kickoffExpertId) return;
    if (
      !hydratedMessages.some((m) => getKickoffExpertId(m) === kickoffExpertId)
    )
      return;
    useCopilotStreamStore.getState().updateCoord(sessionId, {
      lastSubmittedMessageText: null,
      lastSubmittedKickoffExpertId: null,
      lastSubmittedKickoffAttemptToken: null,
    });
  }, [hydratedMessages, sessionId]);

  const notice = snapshot.notice;
  return {
    followBackendTurn: () => runtime?.followBackendTurn(),
    messages,
    appendLocalUserRows: (entries: readonly { id: string; text: string }[]) =>
      runtime?.appendLocalUserRows(entries),
    /** Follow-ups drained into the running turns, for the chip strip. */
    drainedCount: drainedRowCount(snapshot),
    sendMessage,
    stop,
    status: chatStatus(snapshot.phase),
    error: snapshot.error ?? undefined,
    // The runtime never locks the composer: a send while it resumes is queued.
    isReconnecting: false,
    isFinishProbing: false,
    isRestoringActiveSession: notice !== null,
    restoreStatusMessage: notice ? NOTICE_TEXT[notice] : null,
    isSyncing: false,
    isUserStoppingRef,
    isUserStopping: snapshot.stopped,
    turnStats,
    oldestSequence: history.oldestSequence,
    rateLimitMessage,
    platformLimitFailure,
    providerLimit,
    dismissProviderLimit: () => {
      if (sessionId && providerLimit) {
        dismissedProviderFailureRef.current = {
          sessionId,
          fingerprint: providerFailureFingerprint(providerLimit),
        };
      }
      setProviderLimit(null);
    },
    dismissRateLimit: () => {
      setRateLimitMessage(null);
      setPlatformLimitFailure(null);
    },
  };
}

/**
 * The rows above the runtime's tail, from every view this mount has seen:
 * the session GET returns a sliding window, and a long turn slides it past
 * rows the user already has on screen.
 */
function useSessionHistory(
  sessionId: string | null,
  view: SessionView | null,
  cut: number | null,
) {
  const retained = useRef(new Map<number, PersistedRow>());
  return useMemo(() => {
    for (const raw of view?.messages ?? []) {
      const row = raw as PersistedRow;
      if (typeof row.sequence === "number")
        retained.current.set(row.sequence, row);
    }
    const rows = [...retained.current.values()]
      .filter((row) => cut === null || (row.sequence ?? 0) < cut)
      .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
    const converted = sessionId
      ? convertChatSessionMessagesToUiMessages(sessionId, rows, {
          isComplete: !view?.active_stream,
        })
      : { messages: [], stats: new Map() as TurnStatsMap };
    return {
      messages: converted.messages,
      stats: converted.stats,
      oldestSequence: rows[0]?.sequence ?? null,
    };
  }, [sessionId, view, cut]);
}

// Before the runtime attaches to a running turn, its persisted rows are
// already the tail's; cutting them here keeps them from flashing as history.
function prospectiveCut(runtime: TurnRuntime | null, view: SessionView | null) {
  const active = view?.active_stream;
  if (!runtime || !view || !active || !runtime.wouldAttach(view)) return null;
  if (active.checkpoint) return active.checkpoint.sequence;
  const sequences = (view.messages ?? [])
    .map((row) => (row as PersistedRow).sequence)
    .filter((s): s is number => typeof s === "number");
  return sequences.length ? Math.max(...sequences) + 1 : 0;
}

function withLatestUser(
  messages: UIMessage[],
  historyStats: TurnStatsMap,
  tailStats: TurnStatsMap,
) {
  const stats: TurnStatsMap = new Map(historyStats);
  tailStats.forEach((value, key) => stats.set(key, value));
  const latestUser = messages.findLast((m) => m.role === "user");
  stats.forEach((value, key) => {
    if (value.isLatestUserMessage && key !== latestUser?.id) {
      stats.set(key, { ...value, isLatestUserMessage: false });
    }
  });
  if (latestUser) {
    stats.set(latestUser.id, {
      ...stats.get(latestUser.id),
      isLatestUserMessage: true,
    });
  }
  return stats;
}

function chatStatus(phase: RuntimePhase) {
  switch (phase) {
    case "connecting":
      return "submitted" as const;
    case "live":
    case "resuming":
      return "streaming" as const;
    case "failed":
      return "error" as const;
    default:
      return "ready" as const;
  }
}

function drainedRowCount(snapshot: RuntimeSnapshot) {
  let count = 0;
  for (const seg of snapshot.segments) {
    if (seg.kind !== "turn") continue;
    count += seg.log.rows.filter((row) => row.role === "user").length;
  }
  return count;
}

function viewOf(data: unknown): SessionView | null {
  const response = data as { status?: number; data?: SessionView } | undefined;
  return response?.status === 200 && response.data ? response.data : null;
}

function textOf(input: { parts: UIMessage["parts"] }) {
  return input.parts.map((p) => (p.type === "text" ? p.text : "")).join("");
}

function isComposerEmpty() {
  const composer = document.getElementById(
    "chat-input",
  ) as HTMLTextAreaElement | null;
  return !composer || composer.value.length === 0;
}

function subscribeNothing() {
  return () => {};
}

function emptySnapshot() {
  return EMPTY_SNAPSHOT;
}
