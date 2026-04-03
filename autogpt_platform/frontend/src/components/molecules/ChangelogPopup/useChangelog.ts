"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AUTO_DISMISS_MS,
  CHANGELOG_BASE_URL,
  CHANGELOG_INDEX_MD_URL,
  STORAGE_KEY,
} from "./changelog-constants";

export interface ChangelogEntry {
  /** e.g. "march-20-march-25-2026" */
  slug: string;
  /** e.g. "March 20 – March 25" */
  dateRange: string;
  /** Short highlights string from the index table */
  highlights: string;
  /** Full URL to the entry on the docs site */
  url: string;
  /** URL to the raw .md for this entry */
  mdUrl: string;
}

/**
 * Parse the changelog index markdown.
 *
 * The index page is a markdown table like:
 * | Date | Highlights |
 * | --- | --- |
 * | [March 20 – March 25](https://agpt.co/docs/platform/changelog/changelog/march-20-march-25-2026) | Import workflows … |
 */
function parseChangelogIndex(md: string): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];

  // Match table rows: | [Date range](url) | Highlights |
  const rowPattern =
    /\|\s*\[([^\]]+)\]\((https?:\/\/[^)]+\/changelog\/changelog\/([a-z0-9-]+))\)\s*\|\s*([^|]+)\|/g;

  let match;
  while ((match = rowPattern.exec(md)) !== null) {
    const [, dateRange, url, slug, highlights] = match;
    entries.push({
      slug,
      dateRange: dateRange.trim(),
      highlights: highlights.trim(),
      url,
      mdUrl: `${CHANGELOG_BASE_URL}/${slug}.md`,
    });
  }

  return entries;
}

interface UseChangelogReturn {
  /** Whether the toast popup is visible */
  isVisible: boolean;
  /** Whether the toast is fading out */
  isFading: boolean;
  /** The latest unseen changelog entry */
  latestEntry: ChangelogEntry | null;
  /** All parsed entries from the index */
  allEntries: ChangelogEntry[];
  /** Raw markdown content of the selected entry (fetched on demand) */
  entryMarkdown: string | null;
  /** Whether entry markdown is loading */
  isLoadingMarkdown: boolean;
  /** Dismiss the toast */
  dismiss: () => void;
  /** Pause auto-dismiss (e.g. on hover) */
  pauseAutoDismiss: () => void;
  /** Resume auto-dismiss */
  resumeAutoDismiss: () => void;
  /** Whether the full changelog modal is open */
  showFullChangelog: boolean;
  /** Open the full changelog modal, optionally loading a specific entry */
  openFullChangelog: (entry?: ChangelogEntry) => void;
  /** Close the full modal */
  closeFullChangelog: () => void;
  /** The currently selected entry in the full modal */
  selectedEntry: ChangelogEntry | null;
  /** Select a different entry in the modal sidebar */
  selectEntry: (entry: ChangelogEntry) => void;
}

export function useChangelog(): UseChangelogReturn {
  const [isVisible, setIsVisible] = useState(false);
  const [isFading, setIsFading] = useState(false);
  const [showFullChangelog, setShowFullChangelog] = useState(false);
  const [latestEntry, setLatestEntry] = useState<ChangelogEntry | null>(null);
  const [allEntries, setAllEntries] = useState<ChangelogEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<ChangelogEntry | null>(
    null,
  );
  const [entryMarkdown, setEntryMarkdown] = useState<string | null>(null);
  const [isLoadingMarkdown, setIsLoadingMarkdown] = useState(false);

  const autoDismissTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fadeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isPaused = useRef(false);

  // --- Timers ---

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
      /* noop */
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

  // --- Markdown fetching ---

  const fetchEntryMarkdown = useCallback(async (entry: ChangelogEntry) => {
    setIsLoadingMarkdown(true);
    setEntryMarkdown(null);
    try {
      const res = await fetch(entry.mdUrl);
      if (res.ok) {
        const md = await res.text();
        // Strip GitBook-specific directives ({% hint %}, {% endhint %}, <figure>, <figcaption>, <details>, <summary>)
        const cleaned = md
          .replace(/\{%.*?%\}/gs, "")
          .replace(/<figure>|<\/figure>/g, "")
          .replace(/<figcaption>.*?<\/figcaption>/gs, "")
          .replace(/<details>/g, "\n---\n")
          .replace(/<\/details>/g, "")
          .replace(/<summary>(.*?)<\/summary>/g, "### $1");
        setEntryMarkdown(cleaned);
      }
    } catch {
      /* fail silently */
    } finally {
      setIsLoadingMarkdown(false);
    }
  }, []);

  // --- Modal controls ---

  const openFullChangelog = useCallback(
    (entry?: ChangelogEntry) => {
      clearTimers();
      setIsVisible(false);
      setIsFading(false);
      const target = entry || latestEntry;
      if (target) {
        setSelectedEntry(target);
        fetchEntryMarkdown(target);
        markAsSeen(target.slug);
      }
      setShowFullChangelog(true);
    },
    [clearTimers, latestEntry, fetchEntryMarkdown, markAsSeen],
  );

  const closeFullChangelog = useCallback(() => {
    setShowFullChangelog(false);
    setEntryMarkdown(null);
    setSelectedEntry(null);
  }, []);

  const selectEntry = useCallback(
    (entry: ChangelogEntry) => {
      setSelectedEntry(entry);
      fetchEntryMarkdown(entry);
    },
    [fetchEntryMarkdown],
  );

  // --- Initial fetch ---

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

        // Check if user has already seen this entry
        try {
          const lastSeen = localStorage.getItem(STORAGE_KEY);
          if (lastSeen === entries[0].slug) return;
        } catch {
          /* show anyway */
        }

        // Small delay to let page settle
        setTimeout(() => {
          if (!cancelled) setIsVisible(true);
        }, 1500);
      })
      .catch(() => {
        /* fail silently — changelog is non-critical */
      });

    return () => {
      cancelled = true;
      clearTimers();
    };
  }, [clearTimers]);

  // Start auto-dismiss when toast becomes visible
  useEffect(() => {
    if (isVisible && !isFading) startAutoDismiss();
  }, [isVisible, isFading, startAutoDismiss]);

  return {
    isVisible,
    isFading,
    latestEntry,
    allEntries,
    entryMarkdown,
    isLoadingMarkdown,
    dismiss,
    pauseAutoDismiss,
    resumeAutoDismiss,
    showFullChangelog,
    openFullChangelog,
    closeFullChangelog,
    selectedEntry,
    selectEntry,
  };
}
