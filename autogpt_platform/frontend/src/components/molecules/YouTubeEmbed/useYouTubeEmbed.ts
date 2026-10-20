"use client";

import { useState } from "react";
import {
  isConsentManagerConfigured,
  openConsentSettings,
} from "@/services/consent/consent";
import {
  useConsent,
  useConsentManagerStatus,
} from "@/services/consent/useConsent";

export function useYouTubeEmbed() {
  const consent = useConsent();
  const managerStatus = useConsentManagerStatus();
  // Without a consent banner (self-hosted, or a blocker stopped it) the
  // viewer can still load one video on purpose after reading the notice.
  const [loadedOnRequest, setLoadedOnRequest] = useState(false);

  const canOpenSettings =
    isConsentManagerConfigured() && managerStatus !== "unavailable";

  function loadVideo() {
    setLoadedOnRequest(true);
  }

  return {
    showPlayer: consent.advertising || loadedOnRequest,
    canOpenSettings,
    settingsReady: managerStatus === "ready",
    openSettings: openConsentSettings,
    loadVideo,
  };
}
