"use client";

import { usePathname } from "next/navigation";
import { useState } from "react";
import { useCookieConsent } from "./useCookieConsent";

export function useCookieConsentBanner() {
  const { consent, isLoaded, handleAcceptAll, handleRejectAll } =
    useCookieConsent();
  const [showSettings, setShowSettings] = useState(false);
  const pathname = usePathname();

  // The public tour demo keeps the banner out of the way — DataFast loads
  // there without the consent gate instead (see SetupAnalytics).
  const isPublicTourPage = pathname?.startsWith("/tour") ?? false;

  const shouldShowBanner =
    isLoaded && !consent.hasConsented && !isPublicTourPage;

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
