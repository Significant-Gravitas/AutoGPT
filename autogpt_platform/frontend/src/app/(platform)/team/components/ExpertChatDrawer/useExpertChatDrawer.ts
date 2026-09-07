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
import * as Sentry from "@sentry/nextjs";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatTarget } from "./helpers";

type UiMessages = UIMessage<unknown, UIDataTypes, UITools>[];

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
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [skipLatest, setSkipLatest] = useState(false);
  const pendingPromptRef = useRef<string | null>(null);

  const { mutateAsync: createSession } = usePostV2CreateSession();

  // Autopilot threads carry no expert id to look up by, so they always
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
      sessionId,
      hydratedMessages,
      hasActiveStream,
      refetchSession: sessionQuery.refetch,
      copilotModel: undefined,
    });

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
    setSkipLatest(false);
    setSessionId(null);
    setMessages([]);
    pendingPromptRef.current = null;
    setSeedToSend(seedPrompt);
  }, [threadKey, seedPrompt, setMessages]);

  const startSessionRef = useRef(startSession);
  startSessionRef.current = startSession;
  useEffect(() => {
    if (!seedToSend) return;
    setSeedToSend(null);
    void startSessionRef.current(seedToSend);
  }, [seedToSend]);

  useEffect(() => {
    if (!sessionId || !pendingPromptRef.current) return;
    const prompt = pendingPromptRef.current;
    pendingPromptRef.current = null;
    sendMessage({ text: prompt });
  }, [sessionId, sendMessage]);

  function startNewThread() {
    setSkipLatest(true);
    setSessionId(null);
    setMessages([]);
    pendingPromptRef.current = null;
  }

  async function startSession(firstMessage: string) {
    if (isCreating || !target) return;
    setIsCreating(true);
    try {
      const response = (await createSession(
        expertId ? { data: { expert_id: expertId } } : { data: null },
      )) as unknown as { status: number; data?: { id?: string } };
      if (response.status !== 200 || !response.data?.id) {
        throw new Error("Failed to create expert chat session");
      }
      pendingPromptRef.current = firstMessage;
      setSessionId(response.data.id);
    } catch (err) {
      Sentry.captureException(err);
      toast({
        variant: "destructive",
        title: "Could not start the chat",
        description: "Please try sending your message again.",
      });
    } finally {
      setIsCreating(false);
    }
  }

  async function onSend(message: string) {
    const trimmed = message.trim();
    if (!trimmed) return;
    if (!sessionId) {
      await startSession(trimmed);
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

  const isResolvingSession = !sessionId && wantsLatest && latestQuery.isLoading;

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
