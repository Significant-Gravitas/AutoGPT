"use client";

import { useEffect, useRef, useState } from "react";

type TTSStatus = "idle" | "playing" | "paused";

/**
 * Preferred voice names ranked by quality.
 * The first match found in the browser's available voices wins.
 */
const PREFERRED_VOICES = [
  // macOS high-quality
  "Samantha",
  "Karen",
  "Daniel",
  "Moira",
  "Tessa",
  // Chrome / Android
  "Google UK English Female",
  "Google UK English Male",
  "Google US English",
  // Edge / Windows OneCore
  "Microsoft Zira",
  "Microsoft David",
  "Microsoft Jenny",
  "Microsoft Aria",
  "Microsoft Guy",
];

/**
 * Name fragments that indicate low-quality / robotic synthesis engines.
 * Matching is case-insensitive on `voice.name`.
 */
const ROBOTIC_VOICE_INDICATORS = [
  "espeak",
  "festival",
  "mbrola",
  "flite",
  "pico",
];

/** Returns true when a voice is likely low-quality / robotic. */
function isLikelyRobotic(voice: SpeechSynthesisVoice): boolean {
  const lower = voice.name.toLowerCase();
  return ROBOTIC_VOICE_INDICATORS.some((ind) => lower.includes(ind));
}

function pickBestVoice(): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();

  // 1. Try preferred voices first (known high-quality)
  for (const name of PREFERRED_VOICES) {
    const match = voices.find((v) => v.name.includes(name));
    if (match) return match;
  }

  // 2. Filter out known robotic / low-quality voices
  const nonRobotic = voices.filter((v) => !isLikelyRobotic(v));
  const candidates = nonRobotic.length > 0 ? nonRobotic : voices;

  // 3. Prefer remote / cloud-backed voices (usually higher quality)
  const remote = candidates.filter((v) => !v.localService);
  if (remote.length > 0) {
    return remote.find((v) => v.lang.startsWith("en")) || remote[0];
  }

  // 4. Best remaining local voice: English default → English → default → first
  return (
    candidates.find((v) => v.default && v.lang.startsWith("en")) ||
    candidates.find((v) => v.lang.startsWith("en")) ||
    candidates.find((v) => v.default) ||
    candidates[0]
  );
}

export function useTextToSpeech(text: string) {
  const [status, setStatus] = useState<TTSStatus>("idle");
  const [isSupported, setIsSupported] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    setIsSupported("speechSynthesis" in window);
    return () => {
      window.speechSynthesis?.cancel();
    };
  }, []);

  // Reset state when text changes (e.g. navigating between messages)
  useEffect(() => {
    window.speechSynthesis?.cancel();
    utteranceRef.current = null;
    setStatus("idle");
  }, [text]);

  function play() {
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    if (status === "paused") {
      window.speechSynthesis.resume();
      setStatus("playing");
      return;
    }

    // Cancel any ongoing speech first
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);

    const voice = pickBestVoice();
    if (voice) utterance.voice = voice;

    utteranceRef.current = utterance;

    utterance.onend = () => {
      setStatus("idle");
      utteranceRef.current = null;
    };

    utterance.onerror = () => {
      setStatus("idle");
      utteranceRef.current = null;
    };

    window.speechSynthesis.speak(utterance);
    setStatus("playing");
  }

  function pause() {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    window.speechSynthesis.pause();
    setStatus("paused");
  }

  function stop() {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    utteranceRef.current = null;
    setStatus("idle");
  }

  function toggle() {
    if (status === "playing") {
      stop();
    } else {
      play();
    }
  }

  return {
    status,
    isSupported,
    play,
    pause,
    stop,
    toggle,
  };
}
