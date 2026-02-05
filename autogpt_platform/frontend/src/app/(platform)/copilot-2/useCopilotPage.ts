import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { parseAsString, useQueryState } from "nuqs";
import {
  getGetV2ListSessionsQueryKey,
  getV2GetSession,
  postV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { convertChatSessionMessagesToUiMessages } from "./helpers/convertChatSessionToUiMessages";
import { useQueryClient } from "@tanstack/react-query";

export function useCopilotPage() {
  const [copied, setCopied] = useState(false);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const hydrationSeq = useRef(0);
  const lastHydratedSessionIdRef = useRef<string | null>(null);
  const createSessionPromiseRef = useRef<Promise<string> | null>(null);
  const queuedFirstMessageRef = useRef<string | null>(null);
  const queuedFirstMessageResolverRef = useRef<(() => void) | null>(null);
  const queryClient = useQueryClient();

  function handleCopySessionId() {
    if (!sessionId) return;
    navigator.clipboard.writeText(sessionId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const transport = useMemo(() => {
    if (!sessionId) return null;
    return new DefaultChatTransport({
      api: `/api/chat/sessions/${sessionId}/stream`,
      prepareSendMessagesRequest: ({ messages }) => {
        const last = messages[messages.length - 1];
        return {
          body: {
            message: last.parts
              ?.map((p) => (p.type === "text" ? p.text : ""))
              .join(""),
            is_user_message: last.role === "user",
            context: null,
          },
        };
      },
    });
  }, [sessionId]);

  const { messages, sendMessage, status, error, setMessages } = useChat({
    id: sessionId ?? undefined,
    transport: transport ?? undefined,
  });

  const messagesRef = useRef(messages);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  async function createSession(): Promise<string> {
    if (sessionId) return sessionId;
    if (createSessionPromiseRef.current) return createSessionPromiseRef.current;

    setIsCreatingSession(true);
    const promise = (async () => {
      const response = await postV2CreateSession({
        body: JSON.stringify({}),
      });
      if (response.status !== 200 || !response.data?.id) {
        throw new Error("Failed to create chat session");
      }
      setSessionId(response.data.id);
      queryClient.invalidateQueries({
        queryKey: getGetV2ListSessionsQueryKey(),
      });
      return response.data.id;
    })();

    createSessionPromiseRef.current = promise;

    try {
      return await promise;
    } finally {
      createSessionPromiseRef.current = null;
      setIsCreatingSession(false);
    }
  }

  useEffect(() => {
    hydrationSeq.current += 1;
    const seq = hydrationSeq.current;
    const controller = new AbortController();

    if (!sessionId) {
      setMessages([]);
      lastHydratedSessionIdRef.current = null;
      return;
    }

    const currentSessionId = sessionId;

    if (lastHydratedSessionIdRef.current !== currentSessionId) {
      setMessages([]);
      lastHydratedSessionIdRef.current = currentSessionId;
    }

    async function hydrate() {
      try {
        const response = await getV2GetSession(currentSessionId, {
          signal: controller.signal,
        });
        if (response.status !== 200) return;

        const uiMessages = convertChatSessionMessagesToUiMessages(
          currentSessionId,
          response.data.messages ?? [],
        );
        if (controller.signal.aborted) return;
        if (hydrationSeq.current !== seq) return;

        const localMessagesCount = messagesRef.current.length;
        const remoteMessagesCount = uiMessages.length;

        if (remoteMessagesCount === 0) return;
        if (localMessagesCount > remoteMessagesCount) return;

        setMessages(uiMessages);
      } catch (error) {
        if ((error as { name?: string } | null)?.name === "AbortError") return;
        console.warn("Failed to hydrate chat session:", error);
      }
    }

    void hydrate();

    return () => controller.abort();
  }, [sessionId, setMessages]);

  useEffect(() => {
    if (!sessionId) return;
    const firstMessage = queuedFirstMessageRef.current;
    if (!firstMessage) return;

    queuedFirstMessageRef.current = null;
    sendMessage({ text: firstMessage });
    queuedFirstMessageResolverRef.current?.();
    queuedFirstMessageResolverRef.current = null;
  }, [sendMessage, sessionId]);

  async function onSend(message: string) {
    const trimmed = message.trim();
    if (!trimmed) return;

    if (sessionId) {
      sendMessage({ text: trimmed });
      return;
    }

    queuedFirstMessageRef.current = trimmed;
    const sentPromise = new Promise<void>((resolve) => {
      queuedFirstMessageResolverRef.current = resolve;
    });

    await createSession();
    await sentPromise;
  }

  return {
    copied,
    sessionId,
    messages,
    status,
    error,
    isCreatingSession,
    handleCopySessionId,
    createSession,
    onSend,
  };
}
