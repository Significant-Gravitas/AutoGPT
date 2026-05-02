import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";

interface Args {
  onSend: (message: string) => void;
  disabled?: boolean;
  /** Allow sending when text is empty (e.g. when files are attached). */
  canSendEmpty?: boolean;
  inputId?: string;
}

export function useChatInput({
  onSend,
  disabled = false,
  canSendEmpty = false,
  inputId = "chat-input",
}: Args) {
  const [value, setValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  // Synchronous guard against double-submit — refs update immediately,
  // unlike state which batches and can leave a gap for a second call.
  const isSubmittingRef = useRef(false);
  const { initialPrompt, setInitialPrompt } = useCopilotUIStore();

  useEffect(
    function consumeInitialPrompt() {
      if (!initialPrompt) return;
      setValue((prev) => (prev.length === 0 ? initialPrompt : prev));
      setInitialPrompt(null);
    },
    [initialPrompt, setInitialPrompt],
  );

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

  async function handleSend(message = value) {
    const trimmedMessage = message.trim();
    if (disabled || isSending || (!trimmedMessage && !canSendEmpty)) return;
    if (isSubmittingRef.current) return;

    isSubmittingRef.current = true;
    setIsSending(true);
    try {
      await onSend(trimmedMessage);
      setValue("");
    } finally {
      isSubmittingRef.current = false;
      setIsSending(false);
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const message = formData.get("message");
    void handleSend(typeof message === "string" ? message : value);
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
