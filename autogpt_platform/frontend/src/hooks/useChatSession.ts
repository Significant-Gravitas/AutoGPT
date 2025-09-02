import { useState, useEffect, useCallback, useMemo } from "react";
import { ChatAPI, ChatSession, ChatMessage } from "@/lib/autogpt-server-api/chat";
import BackendAPI from "@/lib/autogpt-server-api";

interface UseChatSessionResult {
  session: ChatSession | null;
  messages: ChatMessage[];
  isLoading: boolean;
  error: Error | null;
  createSession: (systemPrompt?: string) => Promise<void>;
  loadSession: (sessionId: string) => Promise<void>;
  deleteSession: () => Promise<void>;
  clearSession: () => void;
}

export function useChatSession(): UseChatSessionResult {
  const [session, setSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const api = useMemo(() => new BackendAPI(), []);
  const chatAPI = useMemo(() => api.chat, [api]);

  // Load session from localStorage on mount
  useEffect(() => {
    // Check for pending session (from auth redirect)
    const pendingSessionId = localStorage.getItem("pending_chat_session");
    if (pendingSessionId) {
      loadSession(pendingSessionId);
      // Clear the pending session flag
      localStorage.removeItem("pending_chat_session");
      localStorage.setItem("chat_session_id", pendingSessionId);
      return;
    }
    
    // Otherwise check for regular stored session
    const storedSessionId = localStorage.getItem("chat_session_id");
    if (storedSessionId) {
      loadSession(storedSessionId);
    }
  }, []);

  const createSession = useCallback(async (systemPrompt?: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const newSession = await chatAPI.createSession({
        system_prompt: systemPrompt || "You are a helpful assistant that helps users discover and set up AI agents from the AutoGPT marketplace. Be conversational, friendly, and guide users through finding the right agent for their needs.",
      });
      
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

  const loadSession = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const loadedSession = await chatAPI.getSession(sessionId, true);
      setSession(loadedSession);
      setMessages(loadedSession.messages || []);
      
      // Update localStorage
      localStorage.setItem("chat_session_id", sessionId);
    } catch (err) {
      console.error("Failed to load chat session:", err);
      
      // If session doesn't exist, clear localStorage and create a new one
      localStorage.removeItem("chat_session_id");
      
      // Create a new session instead
      console.log("Session not found, creating a new one...");
      try {
        const newSession = await chatAPI.createSession({
          system_prompt: "You are a helpful assistant that helps users discover and set up AI agents from the AutoGPT marketplace. Be conversational, friendly, and guide users through finding the right agent for their needs.",
        });
        
        setSession(newSession);
        setMessages(newSession.messages || []);
        localStorage.setItem("chat_session_id", newSession.id);
      } catch (createErr) {
        setError(createErr as Error);
        console.error("Failed to create new session:", createErr);
      }
    } finally {
      setIsLoading(false);
    }
  }, [chatAPI]);

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

  const clearSession = useCallback(() => {
    setSession(null);
    setMessages([]);
    setError(null);
    localStorage.removeItem("chat_session_id");
  }, []);

  const addMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const updateLastMessage = useCallback((content: string) => {
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
    deleteSession,
    clearSession,
  };
}