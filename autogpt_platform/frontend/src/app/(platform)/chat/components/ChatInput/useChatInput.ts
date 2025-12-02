import { KeyboardEvent, useCallback, useState, useRef, useEffect } from "react";

interface UseChatInputArgs {
  onSend: (message: string) => void;
  disabled?: boolean;
  maxRows?: number;
}

export function useChatInput({
  onSend,
  disabled = false,
  maxRows = 5,
}: UseChatInputArgs) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
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
  }, [value, maxRows]);

  const handleSend = useCallback(() => {
    if (disabled || !value.trim()) return;
    onSend(value.trim());
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, onSend, disabled]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
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
