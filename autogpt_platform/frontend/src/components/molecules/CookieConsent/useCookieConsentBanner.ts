"use client";

import { useConsent } from "@/lib/consent";
import { useState } from "react";

export function useCookieConsentBanner() {
  const { consent, isLoaded, acceptAll, rejectAll } = useConsent();
  const [showSettings, setShowSettings] = useState(false);

  // Don't show banner if user has already consented or if not loaded yet
  const shouldShowBanner = isLoaded && !consent.hasConsented;

  function handleAcceptAll() {
    acceptAll();
  }

  function handleRejectAll() {
    rejectAll();
  }

  function handleOpenSettings() {
    setShowSettings(true);
  }

  function handleCloseSettings() {
    setShowSettings(false);
  }

  return {
    shouldShowBanner,
    showSettings,
    handleAcceptAll,
    handleRejectAll,
    handleOpenSettings,
    handleCloseSettings,
  };
}
