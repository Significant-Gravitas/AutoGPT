import { useEffect, useRef, useCallback } from "react";

interface UseMessageListArgs {
  messageCount: number;
  isStreaming: boolean;
}

interface UseMessageListResult {
  messagesEndRef: React.RefObject<HTMLDivElement>;
  messagesContainerRef: React.RefObject<HTMLDivElement>;
  scrollToBottom: () => void;
}

export function useMessageList({
  messageCount,
  isStreaming,
}: UseMessageListArgs): UseMessageListResult {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(function scrollToBottom() {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  // Auto-scroll when new messages arrive or streaming updates
  useEffect(
    function autoScroll() {
      scrollToBottom();
    },
    [messageCount, isStreaming, scrollToBottom],
  );

  return {
    messagesEndRef,
    messagesContainerRef,
    scrollToBottom,
  };
}
