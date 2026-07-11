"use client";

import { useEffect, useRef, useState } from "react";
import {
  AUTO_DISMISS_MS,
  CHANGELOG_INDEX_MD_URL,
  STORAGE_KEY,
} from "./changelog-constants";
import { ChangelogEntry, parseChangelogIndex } from "./helpers";

export function useChangelog() {
  const [isVisible, setIsVisible] = useState(false);
  const [isFading, setIsFading] = useState(false);
  const [showFullChangelog, setShowFullChangelog] = useState(false);
  const [latestEntry, setLatestEntry] = useState<ChangelogEntry | null>(null);
  const [allEntries, setAllEntries] = useState<ChangelogEntry[]>([]);

  const autoDismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fadeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const revealTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isPaused = useRef(false);
  const isDismissing = useRef(false);
  // Track how much of the auto-dismiss countdown is left so hover/focus can
  // pause and resume the *remaining* time instead of restarting from scratch.
  const dismissStartedAt = useRef(0);
  const remainingMs = useRef(AUTO_DISMISS_MS);

  function clearTimers() {
    if (autoDismissTimer.current) clearTimeout(autoDismissTimer.current);
    if (fadeTimer.current) clearTimeout(fadeTimer.current);
    autoDismissTimer.current = null;
    fadeTimer.current = null;
  }

  function markAsSeen(slug: string) {
    try {
      localStorage.setItem(STORAGE_KEY, slug);
    } catch {
      /* noop */
    }
  }

  function dismiss() {
    if (isDismissing.current) return;
    isDismissing.current = true;
    clearTimers();
    setIsFading(true);
    fadeTimer.current = setTimeout(() => {
      setIsVisible(false);
      setIsFading(false);
      isDismissing.current = false;
      if (latestEntry) markAsSeen(latestEntry.slug);
    }, 500);
  }

  function startAutoDismiss(duration = AUTO_DISMISS_MS) {
    if (isPaused.current || showFullChangelog) return;
    clearTimers();
    dismissStartedAt.current = Date.now();
    remainingMs.current = duration;
    autoDismissTimer.current = setTimeout(() => {
      if (!isPaused.current && !showFullChangelog) dismiss();
    }, duration);
  }

  function pauseAutoDismiss() {
    isPaused.current = true;
    if (autoDismissTimer.current) {
      clearTimeout(autoDismissTimer.current);
      autoDismissTimer.current = null;
      const elapsed = Date.now() - dismissStartedAt.current;
      remainingMs.current = Math.max(0, remainingMs.current - elapsed);
    }
  }

  function resumeAutoDismiss() {
    if (isDismissing.current) return;
    isPaused.current = false;
    // Resume the leftover time rather than restarting the full countdown.
    startAutoDismiss(remainingMs.current);
  }

  function openFullChangelog() {
    clearTimers();
    isPaused.current = true;
    setIsVisible(false);
    setIsFading(false);
    isDismissing.current = false;
    if (latestEntry) markAsSeen(latestEntry.slug);
    setShowFullChangelog(true);
  }

  function closeFullChangelog() {
    setShowFullChangelog(false);
  }

  useEffect(() => {
    let cancelled = false;

    fetch(CHANGELOG_INDEX_MD_URL)
      .then((res) => (res.ok ? res.text() : ""))
      .then((md) => {
        if (cancelled || !md) return;

        const entries = parseChangelogIndex(md);
        if (entries.length === 0) return;

        setAllEntries(entries);
        setLatestEntry(entries[0]);

        try {
          const lastSeen = localStorage.getItem(STORAGE_KEY);
          if (lastSeen === entries[0].slug) return;
        } catch {
          /* show anyway */
        }

        revealTimer.current = setTimeout(() => {
          if (!cancelled) setIsVisible(true);
        }, 1500);
      })
      .catch(() => {
        /* non-critical */
      });

    return () => {
      cancelled = true;
      clearTimers();
      if (revealTimer.current) clearTimeout(revealTimer.current);
    };
  }, []);

  useEffect(() => {
    if (isVisible && !isFading && !showFullChangelog) startAutoDismiss();
  }, [isVisible, isFading, showFullChangelog]);

  return {
    isVisible,
    isFading,
    latestEntry,
    allEntries,
    dismiss,
    pauseAutoDismiss,
    resumeAutoDismiss,
    showFullChangelog,
    openFullChangelog,
    closeFullChangelog,
  };
}
