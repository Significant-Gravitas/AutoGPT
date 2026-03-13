import { ChangeEvent, FormEvent, useEffect, useState } from "react";

interface Args {
  onSend: (message: string) => void;
  disabled?: boolean;
  /** Allow sending when text is empty (e.g. when files are attached). */
  canSendEmpty?: boolean;
  inputId?: string;
  /** Pre-fill the input with this value on mount. */
  initialValue?: string;
}

export function useChatInput({
  onSend,
  disabled = false,
  canSendEmpty = false,
  inputId = "chat-input",
  initialValue,
}: Args) {
  const [value, setValue] = useState(initialValue ?? "");
  const [isSending, setIsSending] = useState(false);

  useEffect(
    function focusOnMount() {
      const textarea = document.getElementById(inputId) as HTMLTextAreaElement;
      if (textarea) textarea.focus();
    },
    [inputId],
  );

  useEffect(
    function focusWhenEnabled() {
      if (disabled) return;
      const textarea = document.getElementById(inputId) as HTMLTextAreaElement;
      if (textarea) textarea.focus();
    },
    [disabled, inputId],
  );

  async function handleSend() {
    if (disabled || isSending || (!value.trim() && !canSendEmpty)) return;

    setIsSending(true);
    try {
      await onSend(value.trim());
      setValue("");
    } finally {
      setIsSending(false);
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    void handleSend();
  }

  function handleChange(e: ChangeEvent<HTMLTextAreaElement>) {
    setValue(e.target.value);
  }

  return {
    value,
    setValue,
    handleSend,
    handleSubmit,
    handleChange,
    isSending,
  };
}
