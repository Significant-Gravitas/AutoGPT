import { useState, useCallback, useRef, useMemo } from "react";
import { StreamChunk } from "@/lib/autogpt-server-api/chat";
import BackendAPI from "@/lib/autogpt-server-api";

interface UseChatStreamResult {
  isStreaming: boolean;
  error: Error | null;
  sendMessage: (sessionId: string, message: string, onChunk?: (chunk: StreamChunk) => void) => Promise<void>;
  stopStreaming: () => void;
}

export function useChatStream(): UseChatStreamResult {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  
  const api = useMemo(() => new BackendAPI(), []);
  const chatAPI = useMemo(() => api.chat, [api]);

  const sendMessage = useCallback(async (
    sessionId: string,
    message: string,
    onChunk?: (chunk: StreamChunk) => void
  ) => {
    setIsStreaming(true);
    setError(null);
    
    // Create new abort controller for this stream
    abortControllerRef.current = new AbortController();
    
    try {
      const stream = chatAPI.streamChat(
        sessionId,
        message,
        "gpt-4o",
        50,
        (err) => {
          setError(err);
          console.error("Stream error:", err);
        }
      );
      
      for await (const chunk of stream) {
        // Check if streaming was aborted
        if (abortControllerRef.current?.signal.aborted) {
          break;
        }
        
        if (onChunk) {
          onChunk(chunk);
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setError(err);
        console.error("Failed to stream message:", err);
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }, [chatAPI]);

  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
    }
  }, []);

  return {
    isStreaming,
    error,
    sendMessage,
    stopStreaming,
  };
}