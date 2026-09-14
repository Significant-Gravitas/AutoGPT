"use client";

import { useEffect, useState } from "react";

import { Key, storage } from "@/services/storage/local-storage";

import { SILENCE_TIMEOUT_MS } from "./useVoiceMode";

const MIN_MS = 2_000;
const MAX_MS = 60_000;

/**
 * How long the mic stays open with nobody talking. No settings UI yet — set
 * `copilot-voice-silence-timeout` (ms) in local storage to change it.
 */
export function useVoiceSilenceTimeout(): number {
  const [timeout, setTimeout] = useState(SILENCE_TIMEOUT_MS);

  useEffect(() => {
    const stored = Number(storage.get(Key.COPILOT_VOICE_SILENCE_TIMEOUT));
    if (stored >= MIN_MS && stored <= MAX_MS) setTimeout(stored);
  }, []);

  return timeout;
}
