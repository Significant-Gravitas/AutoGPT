import { convertChatSessionMessagesToUiMessages } from "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages";
import { queueFollowUpMessage } from "@/app/(platform)/copilot/helpers/queueFollowUpMessage";
import { useCopilotPendingChips } from "@/app/(platform)/copilot/useCopilotPendingChips";
import { useCopilotStream } from "@/app/(platform)/copilot/useCopilotStream";
import {
  useGetV2GetSession,
  usePostV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { MEMORY_CHAT_PROMPTS, type MemoryChatSeed } from "../../helpers";

type UiMessages = UIMessage<unknown, UIDataTypes, UITools>[];

interface Args {
  scopeExpertID: string | null;
}

/**
 * In-pane memory chat, modeled on the builder's chat panel: reuses the
 * copilot chat primitives against a session created here. Every open
 * starts a fresh session for the current scope and auto-sends the seed
 * prompt ("summary" or "forget") as its first message.
 */
export function useMemoryChatPanel({ scopeExpertID }: Args) {
  const [isOpen, setIsOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [startError, setStartError] = useState(false);
  const pendingPromptRef = useRef<string | null>(null);
  // Session-create lock, keyed by scope generation: a stale in-flight create
  // for a previous scope must not block the new scope's first open.
  const openingGenRef = useRef<number | null>(null);
  const lastSeedRef = useRef<MemoryChatSeed | null>(null);

  const { mutateAsync: createSession } = usePostV2CreateSession();

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
      copilotMode: undefined,
      copilotModel: undefined,
    });

  const { queuedMessages, queueMessage } = useCopilotPendingChips({
    sessionId,
    status,
    messages,
    setMessages,
  });

  // The seed fires from an effect so it runs only after the stream hook has
  // re-rendered bound to the just-created session.
  useEffect(() => {
    if (!sessionId || !pendingPromptRef.current) return;
    const prompt = pendingPromptRef.current;
    pendingPromptRef.current = null;
    sendMessage({ text: prompt });
  }, [sessionId, sendMessage]);

  // Switching scope invalidates the open conversation — reset everything and
  // bump the generation so an in-flight session create for the old scope is
  // discarded instead of binding its session under the new scope's name.
  const scopeRef = useRef(scopeExpertID);
  const scopeGenRef = useRef(0);
  useEffect(() => {
    if (scopeRef.current === scopeExpertID) return;
    scopeRef.current = scopeExpertID;
    scopeGenRef.current += 1;
    setIsOpen(false);
    setSessionId(null);
    setMessages([]);
    pendingPromptRef.current = null;
  }, [scopeExpertID, setMessages]);

  async function openWithSeed(seed: MemoryChatSeed) {
    // Latest requested seed wins: a second click during the create-session
    // window updates what gets auto-sent (and what retry re-sends).
    lastSeedRef.current = seed;
    setIsOpen(true);
    setStartError(false);
    if (openingGenRef.current === scopeGenRef.current) return;
    const generation = scopeGenRef.current;
    openingGenRef.current = generation;
    setSessionId(null);
    setMessages([]);
    try {
      const response = (await createSession(
        scopeExpertID ? { data: { expert_id: scopeExpertID } } : { data: null },
      )) as unknown as { status: number; data?: { id?: string } };
      if (response.status !== 200 || !response.data?.id) {
        throw new Error("Failed to create memory chat session");
      }
      if (generation !== scopeGenRef.current) return;
      pendingPromptRef.current = MEMORY_CHAT_PROMPTS[lastSeedRef.current];
      setSessionId(response.data.id);
    } catch (err) {
      Sentry.captureException(err);
      if (generation === scopeGenRef.current) setStartError(true);
    } finally {
      // Only release the lock if this call still owns it — a superseded
      // create must not unlock a newer scope's in-flight create.
      if (openingGenRef.current === generation) openingGenRef.current = null;
    }
  }

  function retryStart() {
    if (lastSeedRef.current) void openWithSeed(lastSeedRef.current);
  }

  function close() {
    setIsOpen(false);
  }

  async function onSend(message: string) {
    const trimmed = message.trim();
    if (!trimmed || !sessionId) return;
    const isInFlight = status === "streaming" || status === "submitted";
    if (isInFlight) {
      try {
        await queueFollowUpMessage(sessionId, trimmed);
        queueMessage(trimmed);
      } catch (err) {
        // The turn finished while we were queueing — send normally instead.
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

  return {
    isOpen,
    openWithSeed,
    close,
    retryStart,
    startError,
    isStarting: isOpen && !sessionId && !startError,
    sessionId,
    messages,
    status,
    error,
    stop,
    onSend,
    queuedMessages,
  };
}
