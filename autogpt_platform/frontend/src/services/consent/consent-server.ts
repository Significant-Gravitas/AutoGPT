import { cookies } from "next/headers";
import {
  isConsentManagerConfigured,
  NO_CONSENT,
  toConsentState,
  type ConsentCategory,
  type ConsentState,
} from "./consent";
import { COOKIEBOT_CONSENT_COOKIE, parseCookieConsent } from "./cookiebot";

interface CookieReader {
  get(name: string): { value: string } | undefined;
}

/** The visitor's consent as sent with the current request, for route handlers and server actions. */
export async function getRequestConsent(): Promise<ConsentState> {
  return readConsentFromCookies(await cookies());
}

export async function requestHasConsentFor(
  category: ConsentCategory,
): Promise<boolean> {
  return (await getRequestConsent())[category];
}

export function readConsentFromCookies(store: CookieReader): ConsentState {
  if (!isConsentManagerConfigured()) return NO_CONSENT;
  return toConsentState(
    parseCookieConsent(store.get(COOKIEBOT_CONSENT_COOKIE)?.value),
  );
}
