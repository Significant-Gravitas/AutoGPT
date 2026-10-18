import { useChatDraftAccess } from "../ChatInput/useChatDraft";
import { SESSION_COMPOSER_ID, sessionDraftKey } from "./helpers";
import { useScheduledSend } from "./useScheduledSend";

interface Args {
  sessionId: string | null;
  onSendNow?: (message: string) => void | Promise<void>;
  onClose: () => void;
}

/** "Send at reset" outside the composer: sets the chat's unsent draft to
 *  send at the reset, or with nothing written yet, opens the composer. */
export function useSendDraftAtReset({ sessionId, onSendNow, onClose }: Args) {
  const scheduled = useScheduledSend({ sessionId, onSendNow });
  const draft = useChatDraftAccess(
    sessionId ? sessionDraftKey(sessionId) : undefined,
  );

  async function sendDraftAtReset() {
    const message = draft.read().trim();
    onClose();
    if (!message) {
      document.getElementById(SESSION_COMPOSER_ID)?.focus();
      return;
    }
    draft.write("");
    await scheduled.scheduleQuietly(message);
  }

  return { sendDraftAtReset, isScheduling: scheduled.isBusy };
}
