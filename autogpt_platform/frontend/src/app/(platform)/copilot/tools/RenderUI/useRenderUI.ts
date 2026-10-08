import { useRef, useState } from "react";
import { BuiltinActionType, type ActionEvent } from "@openuidev/react-lang";
import { getActionFields } from "@/lib/openui/actions";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { buildUIFollowUp, readUIDraft, saveUIDraft } from "./helpers";

export function useRenderUI(
  draftID: string | null,
  source: string,
  title: string,
  readOnly: boolean,
) {
  const { onSend, chatSurface } = useCopilotChatActions();
  const locked = readOnly || chatSurface === "share";
  const key = draftID ? `copilot-ui:${draftID}` : null;
  const [initialState] = useState(() =>
    locked || !key ? {} : readUIDraft(key, source),
  );
  const [view, setView] = useState<"interactive" | "summary">("interactive");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const sending = useRef(false);

  async function send(message: string, fields: Record<string, unknown> = {}) {
    if (locked || sending.current || !message.trim()) return;
    sending.current = true;
    setIsSending(true);
    setSent(false);
    setError(null);
    try {
      await onSend(buildUIFollowUp(message, title, fields));
      setSent(true);
    } catch {
      setError(
        "Your follow-up couldn't be sent. Your inputs are still here; try again.",
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

  function onStateUpdate(state: Record<string, unknown>) {
    if (!locked && key) saveUIDraft(key, source, state);
  }

  return {
    locked,
    initialState,
    view,
    setView,
    isSending,
    error,
    sent,
    send,
    onAction,
    onStateUpdate,
  };
}
