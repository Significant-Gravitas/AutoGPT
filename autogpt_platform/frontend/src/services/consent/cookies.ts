import * as Sentry from "@sentry/nextjs";
import { Key, storage } from "../storage/local-storage";
import {
  ANALYTICS_CONSENT_COOKIE,
  ANALYTICS_CONSENT_GRANTED,
} from "./constants";

const CONSENT_COOKIE_MAX_AGE_SECONDS = 365 * 24 * 60 * 60;

export interface ConsentPreferences {
  hasConsented: boolean;
  timestamp: number;
  analytics: boolean;
  monitoring: boolean;
}

export const DEFAULT_CONSENT: ConsentPreferences = {
  hasConsented: false,
  timestamp: Date.now(),
  analytics: false,
  monitoring: false,
};

export const COOKIE_CATEGORIES = {
  essential: {
    name: "Essential Cookies",
    description: "Required for login, authentication, and core functionality",
    alwaysActive: true,
  },
  analytics: {
    name: "Analytics & Performance",
    description:
      "Help us understand how you use AutoGPT to improve our service (Google Analytics, Vercel Analytics, Datafa.st)",
    alwaysActive: false,
  },
  monitoring: {
    name: "Error Monitoring & Session Replay",
    description:
      "Record errors and user sessions to help us fix bugs faster (Sentry - includes screen recording)",
    alwaysActive: false,
  },
} as const;

function load(): ConsentPreferences {
  try {
    const stored = storage.get(Key.COOKIE_CONSENT);
    if (!stored) {
      syncAnalyticsConsentCookie(false);
      return DEFAULT_CONSENT;
    }

    const parsed = JSON.parse(stored) as ConsentPreferences;

    if (
      typeof parsed.hasConsented !== "boolean" ||
      typeof parsed.timestamp !== "number" ||
      typeof parsed.analytics !== "boolean" ||
      typeof parsed.monitoring !== "boolean"
    ) {
      console.warn(
        "Invalid consent data in localStorage, resetting to defaults",
      );
      syncAnalyticsConsentCookie(false);
      return DEFAULT_CONSENT;
    }

    syncAnalyticsConsentCookie(parsed.hasConsented && parsed.analytics);
    return parsed;
  } catch (error) {
    syncAnalyticsConsentCookie(false);
    Sentry.captureException(error);
    console.error("Failed to load consent preferences:", error);
    return DEFAULT_CONSENT;
  }
}

function save(preferences: ConsentPreferences): void {
  try {
    storage.set(Key.COOKIE_CONSENT, JSON.stringify(preferences));
    syncAnalyticsConsentCookie(
      preferences.hasConsented && preferences.analytics,
    );
  } catch (error) {
    Sentry.captureException(error);
    console.error("Failed to save consent preferences:", error);
  }
}

function clear(): void {
  try {
    storage.clean(Key.COOKIE_CONSENT);
    syncAnalyticsConsentCookie(false);
  } catch (error) {
    Sentry.captureException(error);
    console.error("Failed to clear consent preferences:", error);
  }
}

function hasConsented(): boolean {
  const preferences = load();
  return preferences.hasConsented;
}

function hasConsentFor(
  category: keyof Omit<ConsentPreferences, "hasConsented" | "timestamp">,
): boolean {
  const preferences = load();
  return preferences.hasConsented && preferences[category];
}

export const consent = {
  load,
  save,
  clear,
  hasConsented,
  hasConsentFor,
};

function syncAnalyticsConsentCookie(hasConsent: boolean): void {
  if (typeof document === "undefined") return;

  const maxAge = hasConsent ? CONSENT_COOKIE_MAX_AGE_SECONDS : 0;
  const value = hasConsent ? ANALYTICS_CONSENT_GRANTED : "";
  document.cookie = `${ANALYTICS_CONSENT_COOKIE}=${value}; Path=/; Max-Age=${maxAge}; SameSite=Lax${secureCookieAttribute()}`;
}

function secureCookieAttribute(): string {
  return window.location.protocol === "https:" ? "; Secure" : "";
}
