"use client";

import { UIDataTypes, UIMessage, UITools } from "ai";
import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "../../store";

interface UseAutoOpenArtifactsOptions {
  sessionId: string | null;
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  isLoadingSession: boolean;
  isArtifactsEnabled: boolean;
}

/**
 * Manages the auto-open lifecycle for the artifact panel.
 *
 * Auto-open is now **card-based**: each `ArtifactCard` calls
 * `registerArtifactForAutoOpen` on mount — the store decides whether to open
 * the panel. This hook only manages lifecycle:
 * - Resets state on session change
 * - Gates readiness (don't auto-open stale artifacts on session load)
 * - Detects user-close to suppress re-opening
 * - Cleans up on unmount
 *
 * This replaces the previous message-scanning approach which iterated all
 * messages on every streaming render tick.
 */
export function useAutoOpenArtifacts({
  sessionId,
  messages,
  isLoadingSession,
  isArtifactsEnabled,
}: UseAutoOpenArtifactsOptions) {
  const resetArtifactPanel = useCopilotUIStore((s) => s.resetArtifactPanel);
  const resetAutoOpenState = useCopilotUIStore((s) => s.resetAutoOpenState);
  const setAutoOpenReady = useCopilotUIStore((s) => s.setAutoOpenReady);
  const markUserClosedForAutoOpen = useCopilotUIStore(
    (s) => s.markUserClosedForAutoOpen,
  );
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);

  const prevSessionIdRef = useRef(sessionId);
  const wasOpenRef = useRef(false);
  const wasEverLoadingRef = useRef(isLoadingSession);
  const hasMessages = messages.length > 0;

  // Detect the user explicitly closing the panel (isOpen: true → false).
  // Suppress auto-open for the remainder of this session once detected.
  useEffect(() => {
    if (wasOpenRef.current && !isOpen) {
      markUserClosedForAutoOpen();
    }
    wasOpenRef.current = isOpen;
  }, [isOpen, markUserClosedForAutoOpen]);

  // Session change: fully clear the panel and auto-open state so stale
  // artifacts don't bleed into the next chat.
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      resetArtifactPanel();
      resetAutoOpenState();
      wasOpenRef.current = false;
    }
    prevSessionIdRef.current = sessionId;
  }, [sessionId, resetArtifactPanel, resetAutoOpenState]);

  // Mark auto-open as ready once the session is fully loaded.
  //
  // React fires child effects (ArtifactCard) before parent effects (this hook),
  // so all existing cards have already called registerArtifactForAutoOpen by the
  // time this effect fires. Those registrations populate _autoOpenKnownIds
  // without triggering auto-open (ready is still false). Once we set ready=true,
  // only *new* cards mounting in subsequent renders will auto-open.
  //
  // The wasEverLoadingRef guard handles a race where isLoadingSession goes false
  // before messages are hydrated — without it, ready would be set while the
  // messages array is empty, causing all cards that mount during hydration to
  // look "new" and auto-open.
  useEffect(() => {
    if (isLoadingSession) {
      wasEverLoadingRef.current = true;
      return;
    }
    if (!sessionId || !isArtifactsEnabled) return;
    if (wasEverLoadingRef.current && !hasMessages) return;
    setAutoOpenReady();
  }, [
    sessionId,
    isLoadingSession,
    isArtifactsEnabled,
    hasMessages,
    setAutoOpenReady,
  ]);

  // Reset on unmount so navigating away from /copilot can't leave stale state.
  useEffect(() => {
    return () => {
      resetArtifactPanel();
      resetAutoOpenState();
    };
  }, [resetArtifactPanel, resetAutoOpenState]);
}
