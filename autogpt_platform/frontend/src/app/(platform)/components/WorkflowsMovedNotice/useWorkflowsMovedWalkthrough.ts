"use client";

import { useEffect, useRef, useState } from "react";

type PlaybackState = "paused" | "playing" | "ended" | "blocked" | "failed";

export function useWorkflowsMovedWalkthrough() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fallbackRef = useRef<HTMLParagraphElement>(null);
  const focusFallback = useRef(false);
  const [playback, setPlayback] = useState<PlaybackState>("paused");
  const [hasStarted, setHasStarted] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [durationLabel, setDurationLabel] = useState<string | null>(null);

  useEffect(() => {
    if (playback === "failed" && focusFallback.current)
      fallbackRef.current?.focus();
  }, [playback]);

  useEffect(() => {
    const video = videoRef.current;
    return () => video?.pause();
  }, []);

  async function togglePlayback() {
    const video = videoRef.current;
    if (!video || playback === "failed") return;
    if (playback === "playing") {
      video.pause();
      setPlayback("paused");
      return;
    }
    if (isStarting) return;
    if (playback === "ended") video.currentTime = 0;
    setIsStarting(true);
    setPlayback("paused");
    try {
      await video.play();
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      if (videoRef.current === video)
        setPlayback((state) => (state === "failed" ? state : "blocked"));
    } finally {
      if (videoRef.current === video) setIsStarting(false);
    }
  }

  function handleError() {
    focusFallback.current =
      containerRef.current?.contains(document.activeElement) ?? false;
    setPlayback("failed");
  }

  function handlePlay() {
    setHasStarted(true);
    setPlayback((state) => (state === "failed" ? state : "playing"));
  }

  function handleMetadata() {
    const duration = videoRef.current?.duration;
    if (!duration || !Number.isFinite(duration) || duration < 0) return;
    const seconds = Math.ceil(duration);
    setDurationLabel(
      `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`,
    );
  }

  return {
    videoRef,
    containerRef,
    fallbackRef,
    playback,
    hasStarted,
    isStarting,
    durationLabel,
    togglePlayback,
    handleError,
    handlePlay,
    handleMetadata,
    handlePause: () =>
      setPlayback((state) =>
        state === "ended" || state === "failed" ? state : "paused",
      ),
    handleEnded: () =>
      setPlayback((state) => (state === "failed" ? state : "ended")),
  };
}
