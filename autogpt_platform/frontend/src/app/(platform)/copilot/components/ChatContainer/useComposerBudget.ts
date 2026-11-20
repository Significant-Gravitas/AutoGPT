import {
  getResetWait,
  type UsageBudget,
} from "@/services/usageExperience/budget";
import { SEND_AT_RESET_PLACEHOLDER } from "../SendAtReset/helpers";
import { useScheduledSendHandoff } from "../SendAtReset/scheduledSendHandoff";
import { useHandedOffSend } from "../SendAtReset/useHandedOffSend";
import { useScheduledSend } from "../SendAtReset/useScheduledSend";
import { spendsPlatformBudget } from "../usageHelpers";

interface Args {
  sessionId: string | null;
  authProvider?: string | null;
  budget?: UsageBudget | null;
  isLimitReached: boolean;
  /** A lock that waiting for the budget doesn't lift (an archived expert). */
  isOtherwiseLocked: boolean;
  onSend: (message: string) => void | Promise<void>;
  onCreateSession: () => void | Promise<string>;
  inputId: string;
}

export function useComposerBudget({
  sessionId,
  authProvider,
  budget,
  isLimitReached,
  isOtherwiseLocked,
  onSend,
  onCreateSession,
  inputId,
}: Args) {
  const isPlatformChat = spendsPlatformBudget(authProvider);
  const canWaitForReset =
    isLimitReached &&
    !isOtherwiseLocked &&
    isPlatformChat &&
    getResetWait(budget) !== null;
  const scheduled = useScheduledSend({ sessionId, onSendNow: onSend, inputId });
  useHandedOffSend(sessionId, scheduled);

  // A new chat has no session to set the send on: create it, then hand the
  // message to its composer, which sets the send once the page is on it.
  async function scheduleInNewChat(message: string) {
    const created = await onCreateSession();
    if (typeof created !== "string") throw new Error("Couldn’t start a chat");
    useScheduledSendHandoff.getState().hold(created, message);
  }

  return {
    atReset: canWaitForReset
      ? { onSchedule: sessionId ? scheduled.schedule : scheduleInNewChat }
      : undefined,
    placeholder: canWaitForReset ? SEND_AT_RESET_PLACEHOLDER : undefined,
    showLowHint: isPlatformChat && !isLimitReached && budget?.state === "low",
    scheduled,
  };
}
