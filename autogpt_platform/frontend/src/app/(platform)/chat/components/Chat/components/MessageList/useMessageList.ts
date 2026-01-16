import { useEffect, useRef, useCallback } from "react";

interface UseMessageListArgs {
  messageCount: number;
  isStreaming: boolean;
}

export function useMessageList({
  messageCount,
  isStreaming,
}: UseMessageListArgs) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messageCount, isStreaming, scrollToBottom]);

  return {
    messagesEndRef,
    messagesContainerRef,
    scrollToBottom,
  };
}
