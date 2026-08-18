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
  const openingRef = useRef(false);
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

  // Switching scope invalidates the open conversation — reset everything.
  const scopeRef = useRef(scopeExpertID);
  useEffect(() => {
    if (scopeRef.current === scopeExpertID) return;
    scopeRef.current = scopeExpertID;
    setIsOpen(false);
    setSessionId(null);
    setMessages([]);
    pendingPromptRef.current = null;
  }, [scopeExpertID, setMessages]);

  async function openWithSeed(seed: MemoryChatSeed) {
    lastSeedRef.current = seed;
    setIsOpen(true);
    setStartError(false);
    if (openingRef.current) return;
    openingRef.current = true;
    setSessionId(null);
    setMessages([]);
    try {
      const response = (await createSession(
        scopeExpertID ? { data: { expert_id: scopeExpertID } } : { data: null },
      )) as unknown as { status: number; data?: { id?: string } };
      if (response.status !== 200 || !response.data?.id) {
        throw new Error("Failed to create memory chat session");
      }
      pendingPromptRef.current = MEMORY_CHAT_PROMPTS[seed];
      setSessionId(response.data.id);
    } catch (err) {
      Sentry.captureException(err);
      setStartError(true);
    } finally {
      openingRef.current = false;
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
      queueMessage(trimmed);
      try {
        await queueFollowUpMessage(sessionId, trimmed);
      } catch (err) {
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
