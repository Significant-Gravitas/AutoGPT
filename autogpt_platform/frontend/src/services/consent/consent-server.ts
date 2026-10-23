import { headers } from "next/headers";
import {
  isConsentManagerConfigured,
  NO_CONSENT,
  toConsentState,
  type ConsentCategory,
  type ConsentState,
} from "./consent";
import { parseCookieConsentHeader } from "./cookiebot";

/** The visitor's consent as sent with the current request, for route handlers and server actions. */
export async function getRequestConsent(): Promise<ConsentState> {
  // The raw header rather than cookies(): that keeps only the last of two
  // same-named cookies, and the client weighs every copy.
  return readConsentFromCookieHeader((await headers()).get("cookie"));
}

export async function requestHasConsentFor(
  category: ConsentCategory,
): Promise<boolean> {
  return (await getRequestConsent())[category];
}

export function readConsentFromCookieHeader(
  cookieHeader: string | null | undefined,
): ConsentState {
  if (!isConsentManagerConfigured()) return NO_CONSENT;
  return toConsentState(parseCookieConsentHeader(cookieHeader));
}
