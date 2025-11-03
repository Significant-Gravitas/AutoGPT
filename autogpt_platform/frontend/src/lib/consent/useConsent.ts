"use client";

import { useState, useEffect, useCallback } from "react";
import { ConsentPreferences, DEFAULT_CONSENT } from "./types";
import { loadConsent, saveConsent } from "./storage";

/**
 * React hook for managing cookie consent state
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { consent, updateConsent, acceptAll, rejectAll } = useConsent();
 *
 *   if (!consent.hasConsented) {
 *     return <CookieBanner onAcceptAll={acceptAll} onRejectAll={rejectAll} />;
 *   }
 *
 *   return <div>Content</div>;
 * }
 * ```
 */
export function useConsent() {
  const [consent, setConsent] = useState<ConsentPreferences>(DEFAULT_CONSENT);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load consent on mount
  useEffect(() => {
    const stored = loadConsent();
    setConsent(stored);
    setIsLoaded(true);
  }, []);

  /**
   * Update consent preferences
   */
  const updateConsent = useCallback(
    (updates: Partial<ConsentPreferences>) => {
      const newConsent: ConsentPreferences = {
        ...consent,
        ...updates,
        hasConsented: true,
        timestamp: Date.now(),
      };
      setConsent(newConsent);
      saveConsent(newConsent);

      // Trigger a page reload to apply consent changes
      // This ensures analytics scripts are loaded/unloaded appropriately
      if (typeof window !== "undefined") {
        window.location.reload();
      }
    },
    [consent],
  );

  /**
   * Accept all non-essential cookies
   */
  const acceptAll = useCallback(() => {
    updateConsent({
      analytics: true,
      monitoring: true,
    });
  }, [updateConsent]);

  /**
   * Reject all non-essential cookies
   */
  const rejectAll = useCallback(() => {
    updateConsent({
      analytics: false,
      monitoring: false,
    });
  }, [updateConsent]);

  /**
   * Reset consent (for testing or user-initiated reset)
   */
  const resetConsent = useCallback(() => {
    setConsent(DEFAULT_CONSENT);
    saveConsent(DEFAULT_CONSENT);
  }, []);

  return {
    consent,
    isLoaded,
    updateConsent,
    acceptAll,
    rejectAll,
    resetConsent,
  };
}
