"use client";

import { useState, useEffect, useCallback } from "react";
import {
  consent,
  ConsentPreferences,
  DEFAULT_CONSENT,
} from "@/services/consent/cookies";

export function useCookieConsent() {
  const [consentState, setConsentState] =
    useState<ConsentPreferences>(DEFAULT_CONSENT);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const stored = consent.load();
    setConsentState(stored);
    setIsLoaded(true);
  }, []);

  const handleUpdateConsent = useCallback(
    (updates: Partial<ConsentPreferences>) => {
      const newConsent: ConsentPreferences = {
        ...consentState,
        ...updates,
        hasConsented: true,
        timestamp: Date.now(),
      };
      setConsentState(newConsent);
      consent.save(newConsent);

      if (typeof window !== "undefined") {
        window.location.reload();
      }
    },
    [consentState],
  );

  const handleAcceptAll = useCallback(() => {
    handleUpdateConsent({
      analytics: true,
      monitoring: true,
    });
  }, [handleUpdateConsent]);

  const handleRejectAll = useCallback(() => {
    handleUpdateConsent({
      analytics: false,
      monitoring: false,
    });
  }, [handleUpdateConsent]);

  const handleResetConsent = useCallback(() => {
    setConsentState(DEFAULT_CONSENT);
    consent.save(DEFAULT_CONSENT);
  }, []);

  return {
    consent: consentState,
    isLoaded,
    handleUpdateConsent,
    handleAcceptAll,
    handleRejectAll,
    handleResetConsent,
  };
}
