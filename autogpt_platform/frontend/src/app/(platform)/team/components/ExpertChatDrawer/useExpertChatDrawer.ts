import {
  buildKickoffMessage,
  clearKickoffPending,
  type ExpertKickoffMetadata,
  getKickoffStatus,
  type KickoffAttemptToken,
  markKickoffDone,
  markKickoffPending,
  withKickoffLock,
} from "@/app/(platform)/copilot/expertKickoff";
import { convertChatSessionMessagesToUiMessages } from "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages";
import { queueFollowUpMessage } from "@/app/(platform)/copilot/helpers/queueFollowUpMessage";
import { latestExpertSessionParams } from "@/app/(platform)/copilot/expertSessionQuery";
import { useCopilotPendingChips } from "@/app/(platform)/copilot/useCopilotPendingChips";
import { useCopilotStream } from "@/app/(platform)/copilot/useCopilotStream";
import {
  useGetV2GetSession,
  useGetV2ListSessions,
  usePostV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import * as Sentry from "@sentry/nextjs";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatTarget } from "./helpers";

type UiMessages = UIMessage<unknown, UIDataTypes, UITools>[];

interface PendingSend {
  text: string;
  metadata?: ExpertKickoffMetadata;
}

interface KickoffAttempt {
  userId: string;
  expertId: string;
  token: KickoffAttemptToken;
}

interface Args {
  target: ChatTarget | null;
  isOpen: boolean;
  /** Resume the expert's latest thread on open; off = always start fresh. */
  resumeLatest: boolean;
  /** Bump to drop the current thread without remounting the drawer, so an
   *  already-open panel swaps content instead of replaying its animation. */
  threadKey: number;
  /** First message to send in the thread started by `threadKey`. */
  seedPrompt: string | null;
}

export function useExpertChatDrawer({
  target,
  isOpen,
  resumeLatest,
  threadKey,
  seedPrompt,
}: Args) {
  const expertId = target?.expertId ?? null;
  const userId = useAuthStore((state) => state.user?.id) ?? null;
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [skipLatest, setSkipLatest] = useState(false);
  const [kickoffCheckedFor, setKickoffCheckedFor] = useState<string | null>(
    null,
  );
  const pendingPromptRef = useRef<PendingSend | null>(null);
  const pendingAfterKickoffRef = useRef<string | null>(null);
  const kickoffAttemptRef = useRef<KickoffAttempt | null>(null);
  // Every thread reset bumps the generation; a session create that resolves
  // for an older generation is ignored so its prompt never lands in the new
  // thread, and the new thread is free to create its own session.
  const generationRef = useRef(0);
  const creatingGenerationRef = useRef<number | null>(null);

  const { mutateAsync: createSession } = usePostV2CreateSession();

  // Otto threads carry no expert id to look up by, so they always
  // start fresh; expert threads resume the latest one.
  const wantsLatest =
    isOpen && resumeLatest && !!expertId && !sessionId && !skipLatest;
  const latestQuery = useGetV2ListSessions(
    latestExpertSessionParams(expertId),
    { query: { enabled: wantsLatest, refetchOnWindowFocus: false } },
  );

  useEffect(() => {
    if (!wantsLatest || latestQuery.data?.status !== 200) return;
    const latest = latestQuery.data.data.sessions[0];
    if (latest) setSessionId(latest.id);
  }, [latestQuery.data, wantsLatest]);

  // An expert that has never been kicked off opens the thread itself with its
  // onboarding card, as it does on the copilot page after a hire. Any existing
  // thread means that already happened somewhere else, so only remember it.
  const wantsKickoff =
    isOpen &&
    !!expertId &&
    !!userId &&
    !sessionId &&
    !isCreating &&
    kickoffCheckedFor !== expertId &&
    getKickoffStatus(userId, expertId) === "idle";
  const kickoffCheckQuery = useGetV2ListSessions(
    latestExpertSessionParams(expertId),
    { query: { enabled: wantsKickoff, refetchOnWindowFocus: false } },
  );

  const startKickoffRef = useRef(startKickoff);
  startKickoffRef.current = startKickoff;
  const startSessionRef = useRef(startSession);
  startSessionRef.current = startSession;
  useEffect(() => {
    if (!wantsKickoff || !userId || !expertId) return;
    if (kickoffCheckQuery.isFetching) return;
    if (kickoffCheckQuery.data?.status !== 200) return;
    setKickoffCheckedFor(expertId);
    if (kickoffCheckQuery.data.data.sessions.length > 0) {
      void withKickoffLock(userId, expertId, async () => {
        if (getKickoffStatus(userId, expertId) === "idle") {
          markKickoffDone(
            userId,
            expertId,
            markKickoffPending(userId, expertId),
          );
        }
      })
        .then(() => {
          const pending = pendingAfterKickoffRef.current;
          if (!pending) return;
          pendingAfterKickoffRef.current = null;
          void startSessionRef.current({ text: pending });
        })
        .catch(() => undefined);
      return;
    }
    void startKickoffRef.current(userId, expertId);
  }, [
    expertId,
    kickoffCheckQuery.data,
    kickoffCheckQuery.isFetching,
    userId,
    wantsKickoff,
  ]);

  const sessionQuery = useGetV2GetSession(sessionId ?? "", undefined, {
    query: {
      enabled: !!sessionId,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      refetchOnMount: true,
    },
  });
  const hasActiveStream =
    sessionQuery.data?.status === 200
      ? !!sessionQuery.data.data.active_stream
      : false;

  const hydratedMessages = useMemo<UiMessages | undefined>(() => {
    if (sessionQuery.data?.status !== 200 || !sessionId) return undefined;
    return convertChatSessionMessagesToUiMessages(
      sessionId,
      sessionQuery.data.data.messages ?? [],
      { isComplete: !hasActiveStream },
    ).messages as UiMessages;
  }, [sessionQuery.data, sessionId, hasActiveStream]);

  const { messages, setMessages, sendMessage, stop, status, error } =
    useCopilotStream({
      userId,
      sessionId,
      hydratedMessages,
      hasActiveStream,
      refetchSession: sessionQuery.refetch,
      copilotModel: undefined,
    });

  const hasAssistantReply = messages.some(
    (message) => message.role === "assistant",
  );
  useEffect(() => {
    const attempt = kickoffAttemptRef.current;
    if (!attempt || status !== "ready" || !hasAssistantReply) return;
    kickoffAttemptRef.current = null;
    void withKickoffLock(attempt.userId, attempt.expertId, async () => {
      markKickoffDone(attempt.userId, attempt.expertId, attempt.token);
    })
      .catch(() => undefined)
      .then(async () => {
        const pending = pendingAfterKickoffRef.current;
        if (!pending) return;
        pendingAfterKickoffRef.current = null;
        await sendMessage({ text: pending });
      });
  }, [hasAssistantReply, sendMessage, status]);

  useEffect(() => {
    const attempt = kickoffAttemptRef.current;
    if (!error || !attempt) return;
    kickoffAttemptRef.current = null;
    pendingAfterKickoffRef.current = null;
    void withKickoffLock(attempt.userId, attempt.expertId, async () => {
      clearKickoffPending(attempt.userId, attempt.expertId, attempt.token);
    }).catch(() => undefined);
  }, [error]);

  const { queuedMessages, queueMessage } = useCopilotPendingChips({
    sessionId,
    status,
    messages,
    setMessages,
  });

  const threadKeyRef = useRef(threadKey);
  const [seedToSend, setSeedToSend] = useState<string | null>(null);
  useEffect(() => {
    if (threadKeyRef.current === threadKey) return;
    threadKeyRef.current = threadKey;
    generationRef.current += 1;
    creatingGenerationRef.current = null;
    setIsCreating(false);
    setSkipLatest(true);
    setSessionId(null);
    setMessages([]);
    setKickoffCheckedFor(null);
    pendingPromptRef.current = null;
    pendingAfterKickoffRef.current = null;
    kickoffAttemptRef.current = null;
    setSeedToSend(seedPrompt);
  }, [threadKey, seedPrompt, setMessages]);

  useEffect(() => {
    if (!seedToSend) return;
    setSeedToSend(null);
    void startSessionRef.current({ text: seedToSend });
  }, [seedToSend]);

  useEffect(() => {
    if (!sessionId || !pendingPromptRef.current) return;
    const pending = pendingPromptRef.current;
    pendingPromptRef.current = null;
    sendMessage({ text: pending.text, metadata: pending.metadata });
  }, [sessionId, sendMessage]);

  function startNewThread() {
    generationRef.current += 1;
    creatingGenerationRef.current = null;
    setIsCreating(false);
    setSkipLatest(true);
    setSessionId(null);
    setMessages([]);
    setKickoffCheckedFor(null);
    pendingPromptRef.current = null;
    pendingAfterKickoffRef.current = null;
    kickoffAttemptRef.current = null;
  }

  async function startKickoff(ownerId: string, id: string): Promise<boolean> {
    let attemptToken: KickoffAttemptToken | null = null;
    const started = await withKickoffLock(ownerId, id, async () => {
      if (getKickoffStatus(ownerId, id) !== "idle") return false;
      const token = markKickoffPending(ownerId, id);
      attemptToken = token;
      kickoffAttemptRef.current = { userId: ownerId, expertId: id, token };
      if (sessionId) {
        await sendMessage(buildKickoffMessage(id, token));
        return true;
      }
      const created = await startSession(buildKickoffMessage(id, token), {
        expertKickoff: true,
      });
      if (created) return true;
      kickoffAttemptRef.current = null;
      pendingAfterKickoffRef.current = null;
      clearKickoffPending(ownerId, id, token);
      return false;
    }).catch(() => false);
    if (!started && attemptToken) {
      if (kickoffAttemptRef.current?.token === attemptToken) {
        kickoffAttemptRef.current = null;
      }
      pendingAfterKickoffRef.current = null;
      clearKickoffPending(ownerId, id, attemptToken);
    }
    return started ?? false;
  }

  async function startSession(
    firstMessage: PendingSend,
    options?: { expertKickoff?: boolean },
  ): Promise<boolean> {
    const generation = generationRef.current;
    if (creatingGenerationRef.current === generation || !target) return false;
    creatingGenerationRef.current = generation;
    setIsCreating(true);
    try {
      const response = await createSession(
        expertId
          ? {
              data: {
                expert_id: expertId,
                ...(options?.expertKickoff ? { expert_kickoff: true } : {}),
              },
            }
          : { data: null },
      );
      if (generation !== generationRef.current) return false;
      if (response.status !== 200) {
        throw new Error("Failed to create expert chat session");
      }
      pendingPromptRef.current = firstMessage;
      setSessionId(response.data.id);
      return true;
    } catch (err) {
      if (generation !== generationRef.current) return false;
      Sentry.captureException(err);
      toast({
        variant: "destructive",
        title: "Could not start the chat",
        description: "Please try sending your message again.",
      });
      return false;
    } finally {
      if (creatingGenerationRef.current === generation) {
        creatingGenerationRef.current = null;
        setIsCreating(false);
      }
    }
  }

  async function onSend(message: string) {
    const trimmed = message.trim();
    if (!trimmed) return;
    if (!sessionId && (isCheckingKickoff || isCreating)) {
      pendingAfterKickoffRef.current = trimmed;
      return;
    }
    if (
      userId &&
      expertId &&
      kickoffCheckedFor === expertId &&
      getKickoffStatus(userId, expertId) === "idle"
    ) {
      pendingAfterKickoffRef.current = trimmed;
      await startKickoff(userId, expertId);
      return;
    }
    if (!sessionId) {
      await startSession({ text: trimmed });
      return;
    }
    const isInFlight = status === "streaming" || status === "submitted";
    if (isInFlight) {
      try {
        await queueFollowUpMessage(sessionId, trimmed);
        queueMessage(trimmed);
      } catch (err) {
        if (
          err instanceof Error &&
          err.name === "QueueFollowUpNotActiveError"
        ) {
          sendMessage({ text: trimmed });
          return;
        }
        Sentry.captureException(err);
        toast({
          variant: "destructive",
          title: "Could not queue message",
          description: "Please wait for the current response to finish.",
        });
      }
      return;
    }
    sendMessage({ text: trimmed });
  }

  const isCheckingKickoff =
    wantsKickoff &&
    (kickoffCheckQuery.isFetching || !kickoffCheckQuery.isError);
  const isResolvingSession =
    (!sessionId && wantsLatest && latestQuery.isLoading) || isCheckingKickoff;

  return {
    sessionId,
    startNewThread,
    messages,
    status,
    error,
    stop,
    onSend,
    queuedMessages,
    isResolvingSession,
    isCreating,
  };
}
