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
import { getYouTubeLinks } from "./helpers";

export function useYouTubeEmbed(url: string) {
  const consent = useConsent();
  const managerStatus = useConsentManagerStatus();
  // Without a consent banner (self-hosted installs) the viewer can load a
  // video on purpose after reading the notice. The choice covers that one
  // video only, so switching to another asks again.
  const [loadedVideoId, setLoadedVideoId] = useState<string | null>(null);

  const { videoId, embedUrl, watchUrl } = getYouTubeLinks(url);
  // With a banner configured, consent only ever comes from it, even while
  // its script loads or after a blocker stopped it.
  const hasConsentBanner = isConsentManagerConfigured();
  const loadedOnRequest =
    !hasConsentBanner && videoId !== null && loadedVideoId === videoId;

  function loadVideo() {
    setLoadedVideoId(videoId);
  }

  return {
    videoId,
    embedUrl,
    watchUrl,
    showPlayer: consent.advertising || loadedOnRequest,
    canLoadOnce: !hasConsentBanner && videoId !== null,
    // The banner can be opened from the notice (a blocker may have stopped it).
    canAllowHere:
      videoId !== null && hasConsentBanner && managerStatus !== "unavailable",
    settingsReady: managerStatus === "ready",
    openSettings: openConsentSettings,
    loadVideo,
  };
}
