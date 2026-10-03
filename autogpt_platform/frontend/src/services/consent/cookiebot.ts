// Everything Cookiebot-specific lives in this file and the consent module
// around it; the rest of the app only sees `ConsentState`.
// https://www.cookiebot.com/en/developer/

export const COOKIEBOT_CONSENT_COOKIE = "CookieConsent";
export const COOKIEBOT_SCRIPT_URL = "https://consent.cookiebot.com/uc.js";
export const COOKIEBOT_SCRIPT_ID = "Cookiebot";

// OnAccept/OnDecline also fire on page load for a visitor who answered on an
// earlier visit; OnConsentReady fires once the stored state is known.
export const COOKIEBOT_CONSENT_EVENTS = [
  "CookiebotOnConsentReady",
  "CookiebotOnAccept",
  "CookiebotOnDecline",
  "CookiebotOnLoad",
] as const;

// Fired when the banner is about to ask, i.e. the visitor has no answer yet.
// A visitor outside every consent region gets no banner and no such event.
export const COOKIEBOT_DIALOG_EVENTS = [
  "CookiebotOnDialogInit",
  "CookiebotOnDialogDisplay",
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

/**
 * The script tag carries `id="Cookiebot"`, and browsers expose elements by id
 * on `window`, so until uc.js replaces it `window.Cookiebot` is that element.
 * Only an object with the API's methods counts as the loaded consent manager.
 */
export function getCookiebotAPI(): CookiebotAPI | null {
  if (typeof window === "undefined") return null;
  const candidate: unknown = window.Cookiebot;
  if (
    typeof candidate === "object" &&
    candidate !== null &&
    "renew" in candidate &&
    typeof candidate.renew === "function"
  ) {
    return candidate as CookiebotAPI;
  }
  return null;
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
// Newer versions nest more objects inside it (e.g. consentmode:{…}), which can
// repeat the category names, so only top-level keys count.
const KEY_PATTERN = /^\s*([A-Za-z_$][\w$]*)\s*:([\s\S]*)$/;

const CATEGORIES = ["preferences", "statistics", "marketing"] as const;

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

  const entries = readTopLevelEntries(value.slice(1, -1));
  if (!entries) return null;
  if (!CATEGORIES.some((category) => entries.has(category))) return null;

  // A key repeated at the top level only grants when every copy does.
  const granted = (category: (typeof CATEGORIES)[number]) =>
    entries.get(category)?.every((entry) => entry === "true") ?? false;

  return {
    necessary: true,
    preferences: granted("preferences"),
    statistics: granted("statistics"),
    marketing: granted("marketing"),
  };
}

/**
 * Reads every `CookieConsent` entry in a Cookie header or `document.cookie`.
 * A host-only cookie left beside a domain-wide one after a domain change
 * sends both; the client and the server both resolve that to whatever every
 * copy agrees on, so a stale copy can only ever deny.
 */
export function parseCookieConsentHeader(
  cookieHeader: string | null | undefined,
): CookiebotConsent | null {
  if (!cookieHeader) return null;
  const prefix = `${COOKIEBOT_CONSENT_COOKIE}=`;
  const answers = cookieHeader
    .split(";")
    .map((part) => part.trim())
    .filter((part) => part.startsWith(prefix))
    .map((part) => parseCookieConsent(part.slice(prefix.length)));

  if (!answers.some(Boolean)) return null;
  return answers.reduce<CookiebotConsent>(
    (combined, answer) => {
      const next = answer ?? NECESSARY_ONLY;
      return {
        necessary: true,
        preferences: combined.preferences && next.preferences,
        statistics: combined.statistics && next.statistics,
        marketing: combined.marketing && next.marketing,
      };
    },
    { ...ALL_GRANTED },
  );
}

// Splits the object literal's body on the commas outside nested objects,
// arrays and quoted strings. Returns null when the braces or quotes do not
// balance.
function readTopLevelEntries(body: string): Map<string, string[]> | null {
  const parts: string[] = [];
  let depth = 0;
  let quote: string | null = null;
  let start = 0;

  for (let index = 0; index < body.length; index++) {
    const char = body[index];
    if (quote) {
      if (char === "\\") index++;
      else if (char === quote) quote = null;
    } else if (char === "'" || char === '"') {
      quote = char;
    } else if (char === "{" || char === "[") {
      depth++;
    } else if (char === "}" || char === "]") {
      depth--;
      if (depth < 0) return null;
    } else if (char === "," && depth === 0) {
      parts.push(body.slice(start, index));
      start = index + 1;
    }
  }
  if (quote || depth !== 0) return null;
  parts.push(body.slice(start));

  const entries = new Map<string, string[]>();
  for (const part of parts) {
    const match = KEY_PATTERN.exec(part);
    if (!match) continue;
    const [, key, entry] = match;
    entries.set(key, [...(entries.get(key) ?? []), unquote(entry.trim())]);
  }
  return entries;
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
