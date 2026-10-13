import { useRef, useState } from "react";
import { BuiltinActionType, type ActionEvent } from "@openuidev/react-lang";
import { getActionFields } from "@/lib/openui/actions";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { buildUIFollowUp } from "./helpers";
import { useUIDraft } from "./useUIDraft";

export function useRenderUI(
  draftID: string | null,
  source: string,
  title: string,
  readOnly: boolean,
) {
  const { onSend, chatSurface } = useCopilotChatActions();
  const locked = readOnly || chatSurface === "share";
  const draft = useUIDraft(draftID, source, locked);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const sending = useRef(false);

  async function send(message: string, fields: Record<string, unknown> = {}) {
    if (locked || !draft.writable || sending.current || !message.trim()) return;
    sending.current = true;
    setIsSending(true);
    setSent(false);
    setError(null);
    try {
      await onSend(buildUIFollowUp(message, title, fields));
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

  return {
    locked,
    ...draft,
    draft,
    isSending,
    error,
    sent,
    send,
    onAction,
  };
}
