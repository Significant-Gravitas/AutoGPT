import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  formatResetClock,
  formatWeekday,
} from "@/services/usageExperience/budget";

export const SEND_AT_RESET_LABEL = "Send at reset";
export const SCHEDULE_FAILED_TITLE = "Couldn’t schedule your message";
export const SEND_AT_RESET_PLACEHOLDER =
  "Write now, and it sends when your budget resets…";

/** The input id of an open chat's composer. */
export const SESSION_COMPOSER_ID = "chat-input-session";

export function sessionDraftKey(sessionId: string) {
  return `session:${sessionId}`;
}

/** Why the server would not set a send for the reset: the budget is already
 *  back, or it never refills on its own. Null for any other failure. */
export function atResetRefusal(error: unknown) {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  const detail: unknown = error.response?.detail;
  return detail === "not_out_of_budget" || detail === "no_reset"
    ? detail
    : null;
}

export function scheduledSendLabel(runAt: Date | string, now = new Date()) {
  const date = new Date(runAt);
  const clock = formatResetClock(date);
  return date.toDateString() === now.toDateString()
    ? `Sends at ${clock}`
    : `Sends ${formatWeekday(date)} at ${clock}`;
}

/** Puts a message back in front of whatever the composer holds now, keeping
 *  both: a draft written meanwhile must never be lost to it. */
export function returnToDraft(current: string, message: string) {
  if (!current.trim() || current === message) return message;
  return `${message}\n\n${current}`;
}
