import {
  buildKickoffMessage,
  clearKickoffPending,
  getKickoffStatus,
  type KickoffAttemptToken,
  markKickoffDone,
  markKickoffPending,
  withKickoffLock,
} from "@/app/(platform)/copilot/expertKickoff";
import { describeChatError } from "@/app/(platform)/copilot/chatErrorGuidance";
import { convertChatSessionMessagesToUiMessages } from "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages";
import { queueFollowUpMessage } from "@/app/(platform)/copilot/helpers/queueFollowUpMessage";
import {
  hasAttachments,
  validateAttachments,
  type OutgoingMessage,
} from "@/app/(platform)/copilot/helpers/sendWithAttachments";
import {
  buildStoredAttachmentParts,
  type WorkspaceAttachment,
} from "@/app/(platform)/copilot/helpers/workspaceAttachments";
import { latestExpertSessionParams } from "@/app/(platform)/copilot/expertSessionQuery";
import { usePendingOnboardingSession } from "@/app/(platform)/copilot/components/ExpertOnboardingCard/usePendingOnboardingSession";
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
import { textMessage, type ChatTarget } from "./helpers";
import { recoverUnsentMessage, useExpertChatSend } from "./useExpertChatSend";

type UiMessages = UIMessage<unknown, UIDataTypes, UITools>[];

class SessionStartingError extends Error {}

function notifyStartFailed(error: unknown) {
  toast({
    variant: "destructive",
    title: "Could not start the chat",
    description: describeChatError(
      error instanceof Error ? error : new Error(String(error)),
    ).title,
  });
}

interface KickoffAttempt {
  userId: string;
  expertId: string;
  token: KickoffAttemptToken;
}

/** A prompt sent while the kickoff is being decided. Its send settles only
 *  once the prompt went out, so a failed kickoff hands the failure back to
 *  the composer or card that sent it instead of dropping the words. */
interface QueuedSend {
  message: OutgoingMessage;
  resolve: () => void;
  reject: (err: unknown) => void;
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
  const [suppressOnboarding, setSuppressOnboarding] = useState(!!seedPrompt);
  const [kickoffCheckedFor, setKickoffCheckedFor] = useState<string | null>(
    null,
  );
  const pendingPromptRef = useRef<OutgoingMessage | null>(null);
  const queuedBehindKickoffRef = useRef<QueuedSend | null>(null);
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
  const sendQueuedInNewThreadRef = useRef(sendQueuedInNewThread);
  sendQueuedInNewThreadRef.current = sendQueuedInNewThread;
  useEffect(() => {
    if (!wantsKickoff || !userId || !expertId) return;
    if (kickoffCheckQuery.isFetching) return;
    const settled = kickoffCheckQuery.data;
    if (!settled && !kickoffCheckQuery.isError) return;
    if (!settled || settled.status !== 200) {
      // Without the list there is no telling whether the expert was onboarded
      // elsewhere, so this open skips the kickoff rather than risk asking
      // twice; a prompt queued behind it opens a plain thread instead.
      void sendQueuedInNewThreadRef.current();
      return;
    }
    setKickoffCheckedFor(expertId);
    if (settled.data.sessions.length > 0) {
      void withKickoffLock(userId, expertId, async () => {
        if (getKickoffStatus(userId, expertId) === "idle") {
          markKickoffDone(
            userId,
            expertId,
            markKickoffPending(userId, expertId),
          );
        }
      })
        .catch(() => undefined)
        .then(() => sendQueuedInNewThreadRef.current());
      return;
    }
    void startKickoffRef.current(userId, expertId).catch(notifyStartFailed);
  }, [
    expertId,
    kickoffCheckQuery.data,
    kickoffCheckQuery.isError,
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

  const {
    messages,
    setMessages,
    sendMessage,
    stop: stopStream,
    status,
    error,
    isReconnecting,
    isFinishProbing,
  } = useCopilotStream({
    userId,
    sessionId,
    hydratedMessages,
    hasActiveStream,
    refetchSession: sessionQuery.refetch,
    copilotModel: undefined,
  });
  const { pendingSend, hold, release, send, stop } = useExpertChatSend({
    sessionId,
    status,
    isReconnecting,
    isFinishProbing,
    sendMessage,
    stop: stopStream,
  });

  const hasAssistantReply = messages.some(
    (message) => message.role === "assistant",
  );
  useEffect(() => {
    const attempt = kickoffAttemptRef.current;
    if (!attempt || !sessionId || status !== "ready" || !hasAssistantReply)
      return;
    kickoffAttemptRef.current = null;
    void withKickoffLock(attempt.userId, attempt.expertId, async () => {
      markKickoffDone(attempt.userId, attempt.expertId, attempt.token);
    })
      .catch(() => undefined)
      .then(async () => {
        const queued = takeQueued();
        if (!queued) return;
        try {
          await send(sessionId, queued.message);
          queued.resolve();
        } catch (err) {
          queued.reject(err);
        }
      });
  }, [hasAssistantReply, send, sessionId, status]);

  useEffect(() => {
    const attempt = kickoffAttemptRef.current;
    if (!error || !attempt) return;
    kickoffAttemptRef.current = null;
    takeQueued()?.reject(error);
    void withKickoffLock(attempt.userId, attempt.expertId, async () => {
      clearKickoffPending(attempt.userId, attempt.expertId, attempt.token);
    }).catch(() => undefined);
  }, [error]);

  // An empty thread opens the chat whose setup questions still wait, so the
  // answers reach the expert that asked them and its card settles there.
  const isEmptyThread =
    !sessionId ||
    (sessionQuery.data?.status === 200 &&
      messages.length === 0 &&
      status === "ready");
  const pendingOnboarding = usePendingOnboardingSession(
    expertId,
    isOpen &&
      isEmptyThread &&
      !suppressOnboarding &&
      !isCreating &&
      !pendingSend,
  );
  useEffect(() => {
    const pending = pendingOnboarding.sessionId;
    if (!pending || pending === sessionId) return;
    setSessionId(pending);
  }, [pendingOnboarding.sessionId, sessionId]);

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
    takeQueued()?.reject(new Error("The chat was reset"));
    release();
    kickoffAttemptRef.current = null;
    setSeedToSend(seedPrompt);
    setSuppressOnboarding(!!seedPrompt);
  }, [threadKey, seedPrompt, setMessages, release]);

  useEffect(() => {
    if (!seedToSend) return;
    setSeedToSend(null);
    void startSessionRef
      .current(textMessage(seedToSend))
      .catch(notifyStartFailed);
  }, [seedToSend]);

  useEffect(() => {
    if (!sessionId || !pendingPromptRef.current) return;
    const pending = pendingPromptRef.current;
    pendingPromptRef.current = null;
    void send(sessionId, pending).catch((err: unknown) =>
      recoverUnsentMessage(pending, err),
    );
  }, [sessionId, send]);

  function startNewThread() {
    setSuppressOnboarding(false);
    generationRef.current += 1;
    creatingGenerationRef.current = null;
    setIsCreating(false);
    setSkipLatest(true);
    setSessionId(null);
    setMessages([]);
    setKickoffCheckedFor(null);
    pendingPromptRef.current = null;
    takeQueued()?.reject(new Error("The chat was reset"));
    release();
    kickoffAttemptRef.current = null;
  }

  function queueBehindKickoff(message: OutgoingMessage): Promise<void> {
    if (queuedBehindKickoffRef.current) throw new SessionStartingError();
    const settled = new Promise<void>((resolve, reject) => {
      queuedBehindKickoffRef.current = { message, resolve, reject };
    });
    // The sender awaits `settled`; this handler only keeps a rejection that
    // lands before it does from surfacing as unhandled.
    settled.catch(() => undefined);
    return settled;
  }

  function takeQueued(): QueuedSend | null {
    const queued = queuedBehindKickoffRef.current;
    queuedBehindKickoffRef.current = null;
    return queued;
  }

  async function sendQueuedInNewThread() {
    const queued = takeQueued();
    if (!queued) return;
    try {
      const started = await startSession(queued.message);
      if (started) queued.resolve();
      else queued.reject(new Error("The chat was reset"));
    } catch (err) {
      queued.reject(err);
    }
  }

  async function startKickoff(ownerId: string, id: string): Promise<boolean> {
    let attemptToken: KickoffAttemptToken | null = null;
    function clearAttempt() {
      if (!attemptToken) return;
      if (kickoffAttemptRef.current?.token === attemptToken) {
        kickoffAttemptRef.current = null;
      }
      clearKickoffPending(ownerId, id, attemptToken);
    }
    let started: boolean | undefined;
    try {
      started = await withKickoffLock(ownerId, id, async () => {
        if (getKickoffStatus(ownerId, id) !== "idle") return false;
        const token = markKickoffPending(ownerId, id);
        attemptToken = token;
        kickoffAttemptRef.current = { userId: ownerId, expertId: id, token };
        if (sessionId) {
          await sendMessage(buildKickoffMessage(id, token));
          return true;
        }
        const kickoff = buildKickoffMessage(id, token);
        return startSession(textMessage(kickoff.text, kickoff.metadata), {
          expertKickoff: true,
        });
      });
    } catch (err) {
      clearAttempt();
      const queued = takeQueued();
      if (!queued) throw err;
      // The queued sender surfaces the failure and gets its words back.
      queued.reject(err);
      return false;
    }
    if (started) return true;
    // Another tab onboarded this expert first, or the thread was reset
    // mid-create: nothing to wait for, so a queued prompt goes out plainly.
    clearAttempt();
    await sendQueuedInNewThread();
    return false;
  }

  async function startSession(
    firstMessage: OutgoingMessage,
    options?: { expertKickoff?: boolean },
  ): Promise<boolean> {
    const generation = generationRef.current;
    if (!target) return false;
    // A card answered while a typed prompt is still creating the session must
    // not settle on a message that never went out, so the second send rejects.
    if (creatingGenerationRef.current === generation) {
      throw new SessionStartingError();
    }
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
      setSuppressOnboarding(true);
      setSessionId(response.data.id);
      return true;
    } catch (err) {
      if (generation !== generationRef.current) return false;
      Sentry.captureException(err);
      setSuppressOnboarding(false);
      throw err;
    } finally {
      if (creatingGenerationRef.current === generation) {
        creatingGenerationRef.current = null;
        setIsCreating(false);
      }
    }
  }

  async function onSend(
    text: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
  ) {
    const message: OutgoingMessage = {
      text: text.trim(),
      files: files ?? [],
      parts: buildStoredAttachmentParts(workspaceFiles ?? []),
    };
    if (!message.text && !hasAttachments(message)) return;
    if (!validateAttachments(message)) return;
    if (!sessionId || needsKickoff()) {
      // Nothing else draws the message until a session exists to send it to.
      hold(message);
      try {
        await sendBeforeSession(message);
      } catch (err) {
        release(message);
        throw err;
      }
      return;
    }
    if (hasAttachments(message)) {
      void send(sessionId, message).catch((err: unknown) =>
        recoverUnsentMessage(message, err),
      );
      return;
    }
    const isInFlight = status === "streaming" || status === "submitted";
    if (isInFlight) {
      try {
        await queueFollowUpMessage(sessionId, message.text);
        queueMessage(message.text);
      } catch (err) {
        if (
          err instanceof Error &&
          err.name === "QueueFollowUpNotActiveError"
        ) {
          void send(sessionId, message).catch((sendErr: unknown) =>
            recoverUnsentMessage(message, sendErr),
          );
          return;
        }
        Sentry.captureException(err);
        // The composer restores the draft and shows the one toast for it.
        throw err;
      }
      return;
    }
    void send(sessionId, message).catch((err: unknown) =>
      recoverUnsentMessage(message, err),
    );
  }

  function needsKickoff() {
    return (
      !!userId &&
      !!expertId &&
      kickoffCheckedFor === expertId &&
      getKickoffStatus(userId, expertId) === "idle"
    );
  }

  async function sendBeforeSession(message: OutgoingMessage) {
    if (!sessionId && isCheckingKickoff) {
      await queueBehindKickoff(message);
      return;
    }
    if (needsKickoff() && userId && expertId) {
      const sent = queueBehindKickoff(message);
      await startKickoff(userId, expertId);
      await sent;
      return;
    }
    await startSession(message);
  }

  // Cards fail quietly and keep their form, so this path owns the toast; the
  // composer shows its own and restores the draft.
  async function onActionSend(message: string) {
    try {
      await onSend(message);
    } catch (err) {
      if (!(err instanceof SessionStartingError)) notifyStartFailed(err);
      throw err;
    }
  }

  const isCheckingKickoff =
    wantsKickoff &&
    (kickoffCheckQuery.isFetching ||
      (!kickoffCheckQuery.isError && !kickoffCheckQuery.data));
  const isResolvingSession =
    (!sessionId && wantsLatest && latestQuery.isLoading) ||
    isCheckingKickoff ||
    (!sessionId && pendingOnboarding.isChecking);

  return {
    sessionId,
    startNewThread,
    messages,
    status,
    error,
    stop,
    onSend,
    onActionSend,
    queuedMessages,
    isResolvingSession,
    isCreating,
    pendingSend,
  };
}
