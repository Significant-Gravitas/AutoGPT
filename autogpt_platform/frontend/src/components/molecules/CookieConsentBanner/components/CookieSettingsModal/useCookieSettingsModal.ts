"use client";

import { useState, useEffect } from "react";
import { useCookieConsent } from "../../useCookieConsent";

interface Props {
  onClose: () => void;
}

export function useCookieSettingsModal({ onClose }: Props) {
  const { consent, handleUpdateConsent } = useCookieConsent();

  const [analytics, setAnalytics] = useState(consent.analytics);
  const [monitoring, setMonitoring] = useState(consent.monitoring);

  useEffect(() => {
    setAnalytics(consent.analytics);
    setMonitoring(consent.monitoring);
  }, [consent.analytics, consent.monitoring]);

  function handleSavePreferences() {
    handleUpdateConsent({
      analytics,
      monitoring,
    });
    onClose();
  }

  function handleAcceptAll() {
    setAnalytics(true);
    setMonitoring(true);
    handleUpdateConsent({
      analytics: true,
      monitoring: true,
    });
    onClose();
  }

  function handleRejectAll() {
    setAnalytics(false);
    setMonitoring(false);
    handleUpdateConsent({
      analytics: false,
      monitoring: false,
    });
    onClose();
  }

  return {
    analytics,
    setAnalytics,
    monitoring,
    setMonitoring,
    handleSavePreferences,
    handleAcceptAll,
    handleRejectAll,
  };
}
