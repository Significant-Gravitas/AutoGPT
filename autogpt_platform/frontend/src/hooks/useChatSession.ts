import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  // ChatAPI,
  ChatSession,
  ChatMessage,
} from "@/lib/autogpt-server-api/chat";
import BackendAPI from "@/lib/autogpt-server-api";

interface UseChatSessionResult {
  session: ChatSession | null;
  messages: ChatMessage[];
  isLoading: boolean;
  error: Error | null;
  createSession: () => Promise<void>;
  loadSession: (sessionId: string, retryOnFailure?: boolean) => Promise<void>;
  refreshSession: () => Promise<void>;
  deleteSession: () => Promise<void>;
  clearSession: () => void;
}

export function useChatSession(
  urlSessionId?: string | null,
): UseChatSessionResult {
  const [session, setSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const urlSessionIdRef = useRef(urlSessionId);

  const api = useMemo(() => new BackendAPI(), []);
  const chatAPI = useMemo(() => api.chat, [api]);

  // Keep ref updated
  useEffect(() => {
    urlSessionIdRef.current = urlSessionId;
  }, [urlSessionId]);

  const createSession = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const newSession = await chatAPI.createSession({});

      setSession(newSession);
      setMessages(newSession.messages || []);

      // Store session ID in localStorage
      localStorage.setItem("chat_session_id", newSession.id);
    } catch (err) {
      setError(err as Error);
      console.error("Failed to create chat session:", err);
    } finally {
      setIsLoading(false);
    }
  }, [chatAPI]);

  const loadSession = useCallback(
    async (sessionId: string, retryOnFailure = true) => {
      // For URL-based sessions, always try to load (don't skip based on previous failures)
      const failedSessionsKey = "failed_chat_sessions";
      const failedSessions = JSON.parse(
        localStorage.getItem(failedSessionsKey) || "[]",
      );

      // Only skip if it's not explicitly requested via URL (urlSessionId)
      if (
        failedSessions.includes(sessionId) &&
        sessionId !== urlSessionIdRef.current
      ) {
        console.log(
          `Session ${sessionId} previously failed to load, skipping...`,
        );
        // Clear the stored session ID and don't retry
        localStorage.removeItem("chat_session_id");
        if (!session && retryOnFailure) {
          // Create a new session instead
          await createSession();
        }
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const loadedSession = await chatAPI.getSession(sessionId, true);
        console.log("🔍 Loaded session:", sessionId, loadedSession);
        console.log("📝 Messages in session:", loadedSession.messages);
        setSession(loadedSession);
        setMessages(loadedSession.messages || []);

        // Update localStorage to remember this session
        localStorage.setItem("chat_session_id", sessionId);
        // Clear any pending session flag
        localStorage.removeItem("pending_chat_session");

        // Remove from failed sessions if it was there
        const updatedFailedSessions = failedSessions.filter(
          (id: string) => id !== sessionId,
        );
        localStorage.setItem(
          failedSessionsKey,
          JSON.stringify(updatedFailedSessions),
        );
      } catch (err) {
        console.error("Failed to load chat session:", err);

        // Mark this session as failed
        failedSessions.push(sessionId);
        localStorage.setItem(failedSessionsKey, JSON.stringify(failedSessions));

        // If session doesn't exist, clear localStorage
        localStorage.removeItem("chat_session_id");

        if (retryOnFailure) {
          // Create a new session instead
          console.log("Session not found, creating a new one...");
          try {
            const newSession = await chatAPI.createSession({
              system_prompt:
                "You are a helpful assistant that helps users discover and set up AI agents from the AutoGPT marketplace. Be conversational, friendly, and guide users through finding the right agent for their needs.",
            });

            setSession(newSession);
            setMessages(newSession.messages || []);
            localStorage.setItem("chat_session_id", newSession.id);
          } catch (createErr) {
            setError(createErr as Error);
            console.error("Failed to create new session:", createErr);
          }
        }
      } finally {
        setIsLoading(false);
      }
    },
    [chatAPI, createSession, session],
  );

  const deleteSession = useCallback(async () => {
    if (!session) return;

    setIsLoading(true);
    setError(null);

    try {
      await chatAPI.deleteSession(session.id);
      clearSession();
    } catch (err) {
      setError(err as Error);
      console.error("Failed to delete chat session:", err);
    } finally {
      setIsLoading(false);
    }
  }, [session, chatAPI]);

  // Load session from localStorage or URL on mount
  useEffect(() => {
    // If urlSessionId is explicitly null, don't load any session (will create new one)
    if (urlSessionId === null) {
      // Clear stored session to start fresh
      localStorage.removeItem("chat_session_id");
      return;
    }

    // Priority 1: URL session ID (explicit navigation to a session)
    if (urlSessionId) {
      console.log("📍 Loading session from URL:", urlSessionId);
      loadSession(urlSessionId, false); // Don't auto-create new session if URL session fails
      return;
    }

    // Priority 2: Pending session (from auth redirect)
    const pendingSessionId = localStorage.getItem("pending_chat_session");
    if (pendingSessionId) {
      console.log("📍 Loading pending session:", pendingSessionId);
      loadSession(pendingSessionId, false); // Don't retry on failure
      // Clear the pending session flag
      localStorage.removeItem("pending_chat_session");
      localStorage.setItem("chat_session_id", pendingSessionId);
      return;
    }

    // Priority 3: Regular stored session - ONLY load if explicitly in URL
    // Don't automatically load the last session just because it exists in localStorage
    // This prevents unwanted session persistence across page loads
  }, [urlSessionId, loadSession]);

  const refreshSession = useCallback(async () => {
    if (!session) return;

    try {
      console.log("🔄 Refreshing session:", session.id);
      const refreshedSession = await chatAPI.getSession(session.id, true);
      console.log("✅ Refreshed session data:", refreshedSession);
      console.log("📝 Refreshed messages:", refreshedSession.messages);
      setSession(refreshedSession);
      setMessages(refreshedSession.messages || []);
    } catch (err) {
      console.error("Failed to refresh session:", err);
    }
  }, [session, chatAPI]);

  const clearSession = useCallback(() => {
    setSession(null);
    setMessages([]);
    setError(null);
    localStorage.removeItem("chat_session_id");
  }, []);

  const _addMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const _updateLastMessage = useCallback((content: string) => {
    setMessages((prev) => {
      const newMessages = [...prev];
      if (newMessages.length > 0) {
        newMessages[newMessages.length - 1].content = content;
      }
      return newMessages;
    });
  }, []);

  return {
    session,
    messages,
    isLoading,
    error,
    createSession,
    loadSession,
    refreshSession,
    deleteSession,
    clearSession,
  };
}
