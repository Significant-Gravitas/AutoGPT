// Relative imports: jest has no `@/` alias.
import { PLAN_REQUIRED_MESSAGE } from "../chat/chatErrors";

export const CREATE_FAILED = "Couldn't create the link. Try again.";
export const STOP_FAILED = "Couldn't stop sharing. Try again.";

/** The backend sends an absolute URL; a relative one gets the web's origin. */
export function absoluteShareUrl(url: string, frontendUrl: string) {
  if (/^https?:\/\//i.test(url)) return url;
  return `${frontendUrl}${url.startsWith("/") ? "" : "/"}${url}`;
}

/** The share-state route sends the token only; the web builds the same path. */
export function shareUrlForToken(token: string, frontendUrl: string) {
  return `${frontendUrl}/share/chat/${token}`;
}

export function shareErrorText(error: unknown, fallback: string) {
  const status =
    typeof error === "object" && error !== null && "status" in error
      ? error.status
      : null;
  return status === 402 ? PLAN_REQUIRED_MESSAGE : fallback;
}
