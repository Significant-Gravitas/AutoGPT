import { ConsentPreferences, DEFAULT_CONSENT } from "./types";

const STORAGE_KEY = "autogpt_cookie_consent";

/**
 * Load consent preferences from localStorage
 */
export function loadConsent(): ConsentPreferences {
  if (typeof window === "undefined") {
    return DEFAULT_CONSENT;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return DEFAULT_CONSENT;
    }

    const parsed = JSON.parse(stored) as ConsentPreferences;

    // Validate that all required fields exist
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
    console.error("Failed to load consent preferences:", error);
    return DEFAULT_CONSENT;
  }
}

/**
 * Save consent preferences to localStorage
 */
export function saveConsent(preferences: ConsentPreferences): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  } catch (error) {
    console.error("Failed to save consent preferences:", error);
  }
}

/**
 * Clear consent preferences from localStorage
 */
export function clearConsent(): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error("Failed to clear consent preferences:", error);
  }
}

/**
 * Check if user has given consent for a specific category
 */
export function hasConsentFor(
  category: keyof Omit<ConsentPreferences, "hasConsented" | "timestamp">,
): boolean {
  const consent = loadConsent();
  return consent.hasConsented && consent[category];
}
