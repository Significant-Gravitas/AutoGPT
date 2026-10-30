"use client";

import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { useEffect, useRef, useState } from "react";
import { AUTO_DISMISS_MS, STORAGE_KEY } from "./changelog-constants";
import {
  ChangelogEntry,
  fetchChangelogEntries,
  isReleaseBeforeAccount,
} from "./helpers";

export function useChangelog() {
  const accountCreatedAt = useAuthStore((state) => state.user?.created_at);
  const [isVisible, setIsVisible] = useState(false);
  const [isFading, setIsFading] = useState(false);
  const [latestEntry, setLatestEntry] = useState<ChangelogEntry | null>(null);
  // Read at render, so a session that switches to a newer account hides it too.
  const isBeforeAccount =
    latestEntry !== null &&
    isReleaseBeforeAccount(latestEntry, accountCreatedAt);

  const autoDismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fadeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const revealTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pauseSources = useRef(new Set<"hover" | "focus">());
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
    }, 500);
  }

  function startAutoDismiss(duration = AUTO_DISMISS_MS) {
    if (pauseSources.current.size > 0 || isDismissing.current) return;
    clearTimers();
    dismissStartedAt.current = Date.now();
    remainingMs.current = duration;
    autoDismissTimer.current = setTimeout(() => {
      if (pauseSources.current.size === 0) dismiss();
    }, duration);
  }

  function pauseAutoDismiss(source: "hover" | "focus") {
    pauseSources.current.add(source);
    if (autoDismissTimer.current) {
      clearTimeout(autoDismissTimer.current);
      autoDismissTimer.current = null;
      const elapsed = Date.now() - dismissStartedAt.current;
      remainingMs.current = Math.max(0, remainingMs.current - elapsed);
    }
  }

  function resumeAutoDismiss(source: "hover" | "focus") {
    pauseSources.current.delete(source);
    if (isDismissing.current) return;
    // Resume the leftover time rather than restarting the full countdown.
    startAutoDismiss(remainingMs.current);
  }

  useEffect(() => {
    let cancelled = false;

    fetchChangelogEntries()
      .then((entries) => {
        if (cancelled || entries.length === 0) return;

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
    if (!isVisible || isFading || !latestEntry || isBeforeAccount) return;
    markAsSeen(latestEntry.slug);
    startAutoDismiss();
  }, [isVisible, isFading, latestEntry, isBeforeAccount]);

  return {
    isVisible: isVisible && !isBeforeAccount,
    isFading,
    latestEntry,
    dismiss,
    pauseAutoDismiss,
    resumeAutoDismiss,
  };
}
