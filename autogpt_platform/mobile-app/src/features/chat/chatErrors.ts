// Relative import: jest has no `@/` alias, and this module stays import-light
// so its tests load without the auth client or the AI SDK runtime.
import { SignedOutError } from "../../lib/auth/errors";

export const OFFLINE_MESSAGE =
  "Can't reach AutoGPT. Check your connection and try again.";

const FALLBACK_MESSAGE =
  "Otto could not reply just now. Try again in a moment.";

export const PLAN_REQUIRED_MESSAGE =
  "Chat needs an active plan. You can manage your plan on the web.";

const STATUS_MESSAGES: Record<number, string> = {
  402: PLAN_REQUIRED_MESSAGE,
  404: "This chat no longer exists.",
  429: "You have reached your usage limit. Try again later.",
};

const KNOWN_MESSAGES = new Set([
  ...Object.values(STATUS_MESSAGES),
  OFFLINE_MESSAGE,
  FALLBACK_MESSAGE,
  new SignedOutError().message,
]);

/** The sentence for a refused request with this HTTP status. */
export function statusMessage(status: number) {
  return STATUS_MESSAGES[status] ?? FALLBACK_MESSAGE;
}

/** True for the sentence of a request the server refused on purpose. */
export function isStatusMessage(message: string) {
  return Object.values(STATUS_MESSAGES).includes(message);
}

/** The sentence to show for a chat error: ours as is, anything else generic. */
export function chatErrorMessage(error: Error | undefined) {
  if (!error) return null;
  return KNOWN_MESSAGES.has(error.message) ? error.message : FALLBACK_MESSAGE;
}
