"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AUTO_DISMISS_MS,
  CHANGELOG_INDEX_URL,
  STORAGE_KEY,
} from "./changelog-constants";

export interface ChangelogEntry {
  /** URL slug, e.g. "march-20-march-25-2026" */
  slug: string;
  /** Human-readable title parsed from the docs page */
  title: string;
  /** Full URL to the entry on the docs site */
  url: string;
  /** Date range derived from the slug, e.g. "March 20 – March 25, 2026" */
  date: string;
}

/** Parse a slug like "march-20-march-25-2026" into "March 20 – March 25, 2026" */
function slugToDateRange(slug: string): string {
  // Expected format: month-day-month-day-year  or  month-day-month-day-year
  const parts = slug.split("-");
  if (parts.length < 5) return slug;

  // Walk through parts to reconstruct: we need two "month day" pairs and a year
  // e.g. ["february","26","march","4","2026"]
  // or   ["march","20","march","25","2026"]
  // or   ["february","11","february","26","2026"]
  // or   ["january","29","february","11","2026"]
  const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

  // Find year (last 4-digit number)
  const year = parts[parts.length - 1];

  // Everything before the year contains two month-day pairs
  const dateParts = parts.slice(0, -1);

  // Split into two groups: find where the second month name starts
  // Month names are non-numeric strings
  const groups: { month: string; day: string }[] = [];
  let i = 0;
  while (i < dateParts.length) {
    if (isNaN(Number(dateParts[i]))) {
      // This is a month name (could be multi-word but GitBook uses single-word)
      const month = capitalize(dateParts[i]);
      const day = dateParts[i + 1] || "";
      groups.push({ month, day });
      i += 2;
    } else {
      i++;
    }
  }

  if (groups.length === 2) {
    return `${groups[0].month} ${groups[0].day} – ${groups[1].month} ${groups[1].day}, ${year}`;
  }

  return slug;
}

/**
 * Fetch the changelog index page and parse entry links from the HTML.
 * The GitBook page contains <a> tags linking to each entry with their title.
 */
async function fetchChangelogEntries(): Promise<ChangelogEntry[]> {
  const res = await fetch(CHANGELOG_INDEX_URL, {
    next: { revalidate: 3600 }, // Cache for 1 hour
  });

  if (!res.ok) return [];

  const html = await res.text();

  // Parse entries: <a href="/docs/platform/changelog/changelog/SLUG">TITLE</a>
  const pattern =
    /href="(\/docs\/platform\/changelog\/changelog\/([a-z0-9-]+))">([^<]+)<\/a>/g;

  const seen = new Set<string>();
  const entries: ChangelogEntry[] = [];

  let match;
  while ((match = pattern.exec(html)) !== null) {
    const [, path, slug, title] = match;
    if (!seen.has(slug)) {
      seen.add(slug);
      entries.push({
        slug,
        title: title.trim(),
        url: `https://agpt.co${path}`,
        date: slugToDateRange(slug),
      });
    }
  }

  return entries;
}

interface UseChangelogReturn {
  isVisible: boolean;
  isFading: boolean;
  latestEntry: ChangelogEntry | null;
  allEntries: ChangelogEntry[];
  dismiss: () => void;
  pauseAutoDismiss: () => void;
  resumeAutoDismiss: () => void;
  showFullChangelog: boolean;
  setShowFullChangelog: (show: boolean) => void;
}

export function useChangelog(): UseChangelogReturn {
  const [isVisible, setIsVisible] = useState(false);
  const [isFading, setIsFading] = useState(false);
  const [showFullChangelog, setShowFullChangelog] = useState(false);
  const [latestEntry, setLatestEntry] = useState<ChangelogEntry | null>(null);
  const [allEntries, setAllEntries] = useState<ChangelogEntry[]>([]);
  const autoDismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fadeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isPaused = useRef(false);

  const clearTimers = useCallback(() => {
    if (autoDismissTimer.current) clearTimeout(autoDismissTimer.current);
    if (fadeTimer.current) clearTimeout(fadeTimer.current);
    autoDismissTimer.current = null;
    fadeTimer.current = null;
  }, []);

  const markAsSeen = useCallback((slug: string) => {
    try {
      localStorage.setItem(STORAGE_KEY, slug);
    } catch {
      // localStorage unavailable
    }
  }, []);

  const dismiss = useCallback(() => {
    clearTimers();
    setIsFading(true);
    fadeTimer.current = setTimeout(() => {
      setIsVisible(false);
      setIsFading(false);
      if (latestEntry) markAsSeen(latestEntry.slug);
    }, 500);
  }, [clearTimers, latestEntry, markAsSeen]);

  const startAutoDismiss = useCallback(() => {
    if (isPaused.current) return;
    clearTimers();
    autoDismissTimer.current = setTimeout(() => {
      if (!isPaused.current) dismiss();
    }, AUTO_DISMISS_MS);
  }, [clearTimers, dismiss]);

  const pauseAutoDismiss = useCallback(() => {
    isPaused.current = true;
    if (autoDismissTimer.current) {
      clearTimeout(autoDismissTimer.current);
      autoDismissTimer.current = null;
    }
  }, []);

  const resumeAutoDismiss = useCallback(() => {
    isPaused.current = false;
    startAutoDismiss();
  }, [startAutoDismiss]);

  useEffect(() => {
    let cancelled = false;

    fetchChangelogEntries().then((entries) => {
      if (cancelled || entries.length === 0) return;

      const latest = entries[0];
      setAllEntries(entries);
      setLatestEntry(latest);

      // Check if user has already seen this entry
      try {
        const lastSeen = localStorage.getItem(STORAGE_KEY);
        if (lastSeen === latest.slug) return;
      } catch {
        // Show anyway if localStorage unavailable
      }

      // Delay to let the page settle
      setTimeout(() => {
        if (!cancelled) {
          setIsVisible(true);
        }
      }, 1500);
    });

    return () => {
      cancelled = true;
      clearTimers();
    };
  }, [clearTimers]);

  // Start auto-dismiss when popup becomes visible
  useEffect(() => {
    if (isVisible && !isFading) {
      startAutoDismiss();
    }
  }, [isVisible, isFading, startAutoDismiss]);

  return {
    isVisible,
    isFading,
    latestEntry,
    allEntries,
    dismiss,
    pauseAutoDismiss,
    resumeAutoDismiss,
    showFullChangelog,
    setShowFullChangelog,
  };
}
