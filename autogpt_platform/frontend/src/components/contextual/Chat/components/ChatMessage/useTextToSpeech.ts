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
  // Chrome / Android
  "Google UK English Female",
  "Google UK English Male",
  "Google US English",
  // Edge / Windows
  "Microsoft Zira",
  "Microsoft David",
];

function pickBestVoice(): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  for (const name of PREFERRED_VOICES) {
    const match = voices.find((v) => v.name.includes(name));
    if (match) return match;
  }
  // Fallback: prefer any voice flagged as default, or the first English voice
  return (
    voices.find((v) => v.default) ||
    voices.find((v) => v.lang.startsWith("en")) ||
    voices[0]
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
