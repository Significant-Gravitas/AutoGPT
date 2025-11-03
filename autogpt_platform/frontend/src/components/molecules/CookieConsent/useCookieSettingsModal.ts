"use client";

import { useConsent } from "@/lib/consent";
import { useState, useEffect } from "react";

interface Props {
  onClose: () => void;
}

export function useCookieSettingsModal({ onClose }: Props) {
  const { consent, updateConsent } = useConsent();

  // Local state for toggle switches (before saving)
  const [analytics, setAnalytics] = useState(consent.analytics);
  const [monitoring, setMonitoring] = useState(consent.monitoring);

  // Update local state if consent changes
  useEffect(() => {
    setAnalytics(consent.analytics);
    setMonitoring(consent.monitoring);
  }, [consent.analytics, consent.monitoring]);

  function handleSavePreferences() {
    updateConsent({
      analytics,
      monitoring,
    });
    onClose();
  }

  function handleAcceptAll() {
    setAnalytics(true);
    setMonitoring(true);
    updateConsent({
      analytics: true,
      monitoring: true,
    });
    onClose();
  }

  function handleRejectAll() {
    setAnalytics(false);
    setMonitoring(false);
    updateConsent({
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
