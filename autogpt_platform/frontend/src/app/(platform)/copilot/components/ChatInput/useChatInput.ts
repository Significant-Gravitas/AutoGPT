import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { describeSendFailure } from "./helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";

interface Args {
  onSend: (message: string) => void;
  /** Queues text behind the live turn. While `isStreaming`, a submit goes
   *  here instead of `onSend`. */
  onEnqueue?: (message: string) => void | Promise<void>;
  isStreaming?: boolean;
  /** Attachments can't be queued (the pending endpoint takes text only), so
   *  a submit with any attached stays on the `onSend` path. */
  hasAttachments?: boolean;
  disabled?: boolean;
  /** Allow sending when text is empty (e.g. when files are attached). */
  canSendEmpty?: boolean;
  inputId?: string;
}

export function useChatInput({
  onSend,
  onEnqueue,
  isStreaming = false,
  hasAttachments = false,
  disabled = false,
  canSendEmpty = false,
  inputId = "chat-input",
}: Args) {
  const [value, setValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isEnqueueing, setIsEnqueueing] = useState(false);
  // Synchronous guard against double-submit — refs update immediately,
  // unlike state which batches and can leave a gap for a second call.
  const isSubmittingRef = useRef(false);
  const isEnqueueingRef = useRef(false);
  const { initialPrompt, setInitialPrompt, notifyMessageSent } =
    useCopilotUIStore();

  useEffect(
    function consumeInitialPrompt() {
      if (!initialPrompt) return;
      // Guided flows always replace the draft — picking "New skill" after
      // "New scheduled task" must swap the prompt, not keep the stale one.
      setValue(initialPrompt);
      setInitialPrompt(null);
      // Guided flows can prefill while the composer is already mounted
      // (e.g. from a copilot modal) — put the caret in the input so the
      // draft is immediately editable/sendable.
      const textarea = document.getElementById(
        inputId,
      ) as HTMLTextAreaElement | null;
      textarea?.focus();
    },
    [initialPrompt, setInitialPrompt, inputId],
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
    // Clear eagerly: onSend can resolve only when the whole stream finishes,
    // which would leave the sent text sitting in the composer for the entire
    // turn. On failure the draft is restored (see restoreFailedDraft).
    setValue("");
    try {
      await onSend(trimmedMessage);
      notifyMessageSent();
    } catch (error) {
      setValue((current) => restoreFailedDraft(current, message));
      toast({
        title: "Couldn't send message",
        description: describeSendFailure(
          error,
          "your message is back in the composer",
        ),
        variant: "destructive",
      });
    } finally {
      isSubmittingRef.current = false;
      setIsSending(false);
    }
  }

  // Deliberately not behind `isSubmittingRef`: in an existing chat `onSend`
  // settles only when the whole answer has streamed, so that guard is what
  // kept Enter from doing anything while Otto answered. A follow-up typed
  // mid-turn is queued, never sent, so it has a guard of its own.
  async function handleEnqueue(message = value) {
    const trimmedMessage = message.trim();
    if (!onEnqueue || disabled || !trimmedMessage) return;
    if (isEnqueueingRef.current) return;

    isEnqueueingRef.current = true;
    setIsEnqueueing(true);
    setValue("");
    try {
      await onEnqueue(trimmedMessage);
    } catch (error) {
      setValue((current) => restoreFailedDraft(current, message));
      toast({
        title: "Couldn't send message",
        description: describeSendFailure(
          error,
          "your message is back in the composer",
        ),
        variant: "destructive",
      });
    } finally {
      isEnqueueingRef.current = false;
      setIsEnqueueing(false);
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const field = formData.get("message");
    const message = typeof field === "string" ? field : value;
    if (isStreaming && onEnqueue && !hasAttachments) {
      void handleEnqueue(message);
      return;
    }
    void handleSend(message);
  }

  function handleChange(e: ChangeEvent<HTMLTextAreaElement>) {
    setValue(e.target.value);
  }

  return {
    value,
    setValue,
    handleSend,
    handleEnqueue,
    handleSubmit,
    handleChange,
    isSending,
    isEnqueueing,
  };
}

/** A failed send must never cost the user their words. The composer was
 *  cleared eagerly, so the failed message goes back in — and if they started
 *  a new draft during the stream, BOTH are kept: silently picking a winner
 *  deletes the other one for good. */
function restoreFailedDraft(current: string, failed: string) {
  if (!current.trim() || current === failed) return failed;
  return `${failed}\n\n${current}`;
}
