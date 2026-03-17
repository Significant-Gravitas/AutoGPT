"use client";

import { useState } from "react";
import { useCookieConsent } from "./useCookieConsent";

export function useCookieConsentBanner() {
  const { consent, isLoaded, handleAcceptAll, handleRejectAll } =
    useCookieConsent();
  const [showSettings, setShowSettings] = useState(false);

  const shouldShowBanner = isLoaded && !consent.hasConsented;

  function handleAcceptAllClick() {
    handleAcceptAll();
  }

  function handleRejectAllClick() {
    handleRejectAll();
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
    handleAcceptAll: handleAcceptAllClick,
    handleRejectAll: handleRejectAllClick,
    handleOpenSettings,
    handleCloseSettings,
  };
}
