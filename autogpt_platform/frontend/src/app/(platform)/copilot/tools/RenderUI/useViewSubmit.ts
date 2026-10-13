import { useRef, useState } from "react";
import { BuiltinActionType, type ActionEvent } from "@openuidev/react-lang";
import { getActionFields } from "@/lib/openui/actions";
import { buildUIFollowUp } from "./helpers";

export type ViewDeliver = (message: string) => void | Promise<void>;

interface Args {
  title: string;
  disabled: boolean;
  deliver: ViewDeliver | null;
}

export function useViewSubmit({ title, disabled, deliver }: Args) {
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const sending = useRef(false);

  async function send(message: string, fields: Record<string, unknown> = {}) {
    if (disabled || !deliver || sending.current || !message.trim()) return;
    sending.current = true;
    setIsSending(true);
    setSent(false);
    setError(null);
    try {
      await deliver(buildUIFollowUp(message, title, fields));
      setSent(true);
    } catch (cause) {
      setError(
        cause instanceof Error &&
          cause.message.startsWith("This submission exceeds")
          ? cause.message
          : "Your follow-up couldn't be sent. Your inputs are still here; try again.",
      );
    } finally {
      sending.current = false;
      setIsSending(false);
    }
  }

  function onAction(event: ActionEvent) {
    if (event.type !== BuiltinActionType.ContinueConversation) return;
    void send(
      event.humanFriendlyMessage,
      getActionFields(event.formState, event.formName),
    );
  }

  return { isSending, error, sent, send, onAction };
}
