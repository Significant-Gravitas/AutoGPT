import { KeyboardEvent, useCallback, useState, useRef, useEffect } from "react";

interface UseChatInputArgs {
  onSend: (message: string) => void;
  disabled?: boolean;
  maxRows?: number;
}

interface UseChatInputResult {
  value: string;
  setValue: (value: string) => void;
  handleKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  handleSend: () => void;
  textareaRef: React.RefObject<HTMLTextAreaElement>;
}

export function useChatInput({
  onSend,
  disabled = false,
  maxRows = 5,
}: UseChatInputArgs): UseChatInputResult {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(
    function autoResizeTextarea() {
      const textarea = textareaRef.current;
      if (!textarea) return;

      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = "auto";

      // Calculate the number of rows
      const lineHeight = parseInt(
        window.getComputedStyle(textarea).lineHeight,
        10,
      );
      const maxHeight = lineHeight * maxRows;
      const newHeight = Math.min(textarea.scrollHeight, maxHeight);

      textarea.style.height = `${newHeight}px`;
      textarea.style.overflowY =
        textarea.scrollHeight > maxHeight ? "auto" : "hidden";
    },
    [value, maxRows],
  );

  const handleSend = useCallback(
    function handleSend() {
      if (disabled || !value.trim()) return;

      onSend(value.trim());
      setValue("");

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    },
    [value, onSend, disabled],
  );

  const handleKeyDown = useCallback(
    function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
      // Enter without Shift = send message
      // Shift + Enter = new line
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return {
    value,
    setValue,
    handleKeyDown,
    handleSend,
    textareaRef,
  };
}
