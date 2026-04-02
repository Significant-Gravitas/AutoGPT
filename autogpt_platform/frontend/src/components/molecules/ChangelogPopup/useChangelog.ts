"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { changelogEntries, ChangelogEntry } from "./changelog-data";

const STORAGE_KEY = "autogpt-last-seen-changelog";
const AUTO_DISMISS_DELAY = 8000; // 8 seconds before auto-fade

interface UseChangelogReturn {
  /** Whether the popup should be visible */
  isVisible: boolean;
  /** The latest unseen changelog entry */
  latestEntry: ChangelogEntry | null;
  /** All changelog entries */
  allEntries: ChangelogEntry[];
  /** Whether the popup is in its fade-out phase */
  isFading: boolean;
  /** Dismiss the popup and mark current entry as seen */
  dismiss: () => void;
  /** Pause auto-dismiss (e.g. on hover) */
  pauseAutoDismiss: () => void;
  /** Resume auto-dismiss (e.g. on mouse leave) */
  resumeAutoDismiss: () => void;
  /** Open the full changelog view */
  showFullChangelog: boolean;
  /** Toggle full changelog view */
  setShowFullChangelog: (show: boolean) => void;
}

export function useChangelog(): UseChangelogReturn {
  const [isVisible, setIsVisible] = useState(false);
  const [isFading, setIsFading] = useState(false);
  const [showFullChangelog, setShowFullChangelog] = useState(false);
  const [latestEntry, setLatestEntry] = useState<ChangelogEntry | null>(null);
  const autoDismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isPausedRef = useRef(false);

  const clearTimers = useCallback(() => {
    if (autoDismissTimerRef.current) {
      clearTimeout(autoDismissTimerRef.current);
      autoDismissTimerRef.current = null;
    }
    if (fadeTimerRef.current) {
      clearTimeout(fadeTimerRef.current);
      fadeTimerRef.current = null;
    }
  }, []);

  const dismiss = useCallback(() => {
    clearTimers();
    setIsFading(true);

    // After fade animation completes, fully hide
    fadeTimerRef.current = setTimeout(() => {
      setIsVisible(false);
      setIsFading(false);

      // Mark as seen
      if (changelogEntries.length > 0) {
        try {
          localStorage.setItem(STORAGE_KEY, changelogEntries[0].id);
        } catch {
          // localStorage might not be available
        }
      }
    }, 500); // Match CSS transition duration
  }, [clearTimers]);

  const startAutoDismiss = useCallback(() => {
    if (isPausedRef.current) return;

    clearTimers();
    autoDismissTimerRef.current = setTimeout(() => {
      if (!isPausedRef.current) {
        dismiss();
      }
    }, AUTO_DISMISS_DELAY);
  }, [clearTimers, dismiss]);

  const pauseAutoDismiss = useCallback(() => {
    isPausedRef.current = true;
    if (autoDismissTimerRef.current) {
      clearTimeout(autoDismissTimerRef.current);
      autoDismissTimerRef.current = null;
    }
  }, []);

  const resumeAutoDismiss = useCallback(() => {
    isPausedRef.current = false;
    startAutoDismiss();
  }, [startAutoDismiss]);

  useEffect(() => {
    if (changelogEntries.length === 0) return;

    const latest = changelogEntries[0];

    try {
      const lastSeen = localStorage.getItem(STORAGE_KEY);
      if (lastSeen === latest.id) {
        // User has already seen the latest entry
        return;
      }
    } catch {
      // localStorage not available, show the popup anyway
    }

    // Small delay before showing to let the page settle
    const showTimer = setTimeout(() => {
      setLatestEntry(latest);
      setIsVisible(true);
      startAutoDismiss();
    }, 1500);

    return () => {
      clearTimeout(showTimer);
      clearTimers();
    };
  }, [startAutoDismiss, clearTimers]);

  return {
    isVisible,
    latestEntry,
    allEntries: changelogEntries,
    isFading,
    dismiss,
    pauseAutoDismiss,
    resumeAutoDismiss,
    showFullChangelog,
    setShowFullChangelog,
  };
}
