import * as Sentry from "@sentry/nextjs";
import { Key, storage } from "../storage/local-storage";

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
      return DEFAULT_CONSENT;
    }

    return parsed;
  } catch (error) {
    Sentry.captureException(error);
    console.error("Failed to load consent preferences:", error);
    return DEFAULT_CONSENT;
  }
}

function save(preferences: ConsentPreferences): void {
  try {
    storage.set(Key.COOKIE_CONSENT, JSON.stringify(preferences));
  } catch (error) {
    Sentry.captureException(error);
    console.error("Failed to save consent preferences:", error);
  }
}

function clear(): void {
  try {
    storage.clean(Key.COOKIE_CONSENT);
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
