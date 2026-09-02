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
  const [advertising, setAdvertising] = useState(consent.advertising);

  useEffect(() => {
    setAnalytics(consent.analytics);
    setMonitoring(consent.monitoring);
    setAdvertising(consent.advertising);
  }, [consent.analytics, consent.monitoring, consent.advertising]);

  function handleSavePreferences() {
    handleUpdateConsent({
      analytics,
      monitoring,
      advertising,
    });
    onClose();
  }

  function handleAcceptAll() {
    setAnalytics(true);
    setMonitoring(true);
    setAdvertising(true);
    handleUpdateConsent({
      analytics: true,
      monitoring: true,
      advertising: true,
    });
    onClose();
  }

  function handleRejectAll() {
    setAnalytics(false);
    setMonitoring(false);
    setAdvertising(false);
    handleUpdateConsent({
      analytics: false,
      monitoring: false,
      advertising: false,
    });
    onClose();
  }

  return {
    analytics,
    setAnalytics,
    monitoring,
    setMonitoring,
    advertising,
    setAdvertising,
    handleSavePreferences,
    handleAcceptAll,
    handleRejectAll,
  };
}
