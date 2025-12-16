import { KeyboardEvent, useCallback, useEffect, useState } from "react";

interface UseChatInputArgs {
  onSend: (message: string) => void;
  disabled?: boolean;
  maxRows?: number;
  inputId?: string;
}

export function useChatInput({
  onSend,
  disabled = false,
  maxRows = 5,
  inputId = "chat-input",
}: UseChatInputArgs) {
  const [value, setValue] = useState("");

  useEffect(() => {
    const textarea = document.getElementById(inputId) as HTMLTextAreaElement;
    if (!textarea) return;
    textarea.style.height = "auto";
    const lineHeight = parseInt(
      window.getComputedStyle(textarea).lineHeight,
      10,
    );
    const maxHeight = lineHeight * maxRows;
    const newHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = `${newHeight}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [value, maxRows, inputId]);

  const handleSend = useCallback(() => {
    if (disabled || !value.trim()) return;
    onSend(value.trim());
    setValue("");
    const textarea = document.getElementById(inputId) as HTMLTextAreaElement;
    if (textarea) {
      textarea.style.height = "auto";
    }
  }, [value, onSend, disabled, inputId]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleSend();
      }
      // Shift+Enter allows default behavior (new line) - no need to handle explicitly
    },
    [handleSend],
  );

  return {
    value,
    setValue,
    handleKeyDown,
    handleSend,
  };
}
