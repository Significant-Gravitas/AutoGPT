// Everything Cookiebot-specific lives in this file and the consent module
// around it; the rest of the app only sees `ConsentState`.
// https://www.cookiebot.com/en/developer/

export const COOKIEBOT_CONSENT_COOKIE = "CookieConsent";
export const COOKIEBOT_SCRIPT_URL = "https://consent.cookiebot.com/uc.js";

// OnAccept/OnDecline also fire on page load for a visitor who answered on an
// earlier visit; OnConsentReady fires once the stored state is known.
export const COOKIEBOT_CONSENT_EVENTS = [
  "CookiebotOnConsentReady",
  "CookiebotOnAccept",
  "CookiebotOnDecline",
  "CookiebotOnLoad",
] as const;

export interface CookiebotConsent {
  necessary: boolean;
  preferences: boolean;
  statistics: boolean;
  marketing: boolean;
}

interface CookiebotAPI {
  consent: CookiebotConsent;
  consented: boolean;
  declined: boolean;
  hasResponse: boolean;
  renew: () => void;
}

declare global {
  interface Window {
    Cookiebot?: CookiebotAPI;
  }
}

const ALL_GRANTED: CookiebotConsent = {
  necessary: true,
  preferences: true,
  statistics: true,
  marketing: true,
};

const NECESSARY_ONLY: CookiebotConsent = {
  necessary: true,
  preferences: false,
  statistics: false,
  marketing: false,
};

// Cookiebot writes "-1" when the visitor is outside every region its
// configuration requires consent for (all categories accepted), and older
// versions wrote "0" for "declined, necessary cookies only".
const NO_CONSENT_REQUIRED = "-1";
const LEGACY_DECLINED = "0";

// The answer itself is a JavaScript object literal, not JSON, usually
// URL-encoded:
// {stamp:'…',necessary:true,preferences:false,statistics:true,marketing:false,method:'explicit',ver:1,utc:1724770548958,region:'de'}
const ENTRY_PATTERN =
  /([A-Za-z_$][\w$]*)\s*:\s*('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^,}]*)/g;

/**
 * Parses the `CookieConsent` cookie Cookiebot stores the visitor's answer in.
 * Returns null when there is no usable answer, which callers treat as "only
 * necessary cookies".
 */
export function parseCookieConsent(
  raw: string | null | undefined,
): CookiebotConsent | null {
  if (!raw) return null;
  const value = decodeCookieValue(raw).trim();

  if (value === NO_CONSENT_REQUIRED) return ALL_GRANTED;
  if (value === LEGACY_DECLINED) return NECESSARY_ONLY;
  if (!value.startsWith("{") || !value.endsWith("}")) return null;

  const entries = new Map<string, string>();
  for (const [, key, entry] of value.slice(1, -1).matchAll(ENTRY_PATTERN)) {
    entries.set(key, unquote(entry.trim()));
  }

  const categories = ["preferences", "statistics", "marketing"] as const;
  if (!categories.some((category) => entries.has(category))) return null;

  return {
    necessary: true,
    preferences: entries.get("preferences") === "true",
    statistics: entries.get("statistics") === "true",
    marketing: entries.get("marketing") === "true",
  };
}

function unquote(entry: string): string {
  return /^(['"]).*\1$/.test(entry) ? entry.slice(1, -1) : entry;
}

function decodeCookieValue(raw: string): string {
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}
