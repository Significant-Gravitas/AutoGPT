"use client";

import { useEffect, useRef, useState } from "react";

type TTSStatus = "idle" | "playing" | "paused";

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
