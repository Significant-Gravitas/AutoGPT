"use client";

import { useEffect, useRef, useState } from "react";
import {
  AUTO_DISMISS_MS,
  CHANGELOG_BASE_URL,
  CHANGELOG_INDEX_MD_URL,
  STORAGE_KEY,
} from "./changelog-constants";

export interface ChangelogEntry {
  slug: string;
  dateRange: string;
  highlights: string;
  url: string;
  mdUrl: string;
}

function parseChangelogIndex(md: string): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];
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

export function useChangelog() {
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
  const isDismissing = useRef(false);
  const mdAbort = useRef<AbortController | null>(null);

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

  function startAutoDismiss() {
    if (isPaused.current || showFullChangelog) return;
    clearTimers();
    autoDismissTimer.current = setTimeout(() => {
      if (!isPaused.current && !showFullChangelog) dismiss();
    }, AUTO_DISMISS_MS);
  }

  function pauseAutoDismiss() {
    isPaused.current = true;
    if (autoDismissTimer.current) {
      clearTimeout(autoDismissTimer.current);
      autoDismissTimer.current = null;
    }
  }

  function resumeAutoDismiss() {
    if (isDismissing.current) return;
    isPaused.current = false;
    startAutoDismiss();
  }

  function fetchEntryMarkdown(entry: ChangelogEntry) {
    mdAbort.current?.abort();
    const controller = new AbortController();
    mdAbort.current = controller;

    setIsLoadingMarkdown(true);
    setEntryMarkdown(null);

    fetch(entry.mdUrl, { signal: controller.signal })
      .then((res) => (res.ok ? res.text() : ""))
      .then((md) => {
        if (controller.signal.aborted) return;
        const cleaned = md
          .replace(/\{%.*?%\}/gs, "")
          .replace(/<figure>|<\/figure>/g, "")
          .replace(/<figcaption>.*?<\/figcaption>/gs, "")
          .replace(/<details>/g, "\n---\n")
          .replace(/<\/details>/g, "")
          .replace(/<summary>(.*?)<\/summary>/g, "### $1");
        setEntryMarkdown(cleaned);
      })
      .catch(() => {
        /* abort or network error — non-critical */
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoadingMarkdown(false);
      });
  }

  function openFullChangelog(entry?: ChangelogEntry) {
    clearTimers();
    isPaused.current = true;
    setIsVisible(false);
    setIsFading(false);
    isDismissing.current = false;
    const target = entry || latestEntry;
    if (target) {
      setSelectedEntry(target);
      fetchEntryMarkdown(target);
      markAsSeen(target.slug);
    }
    setShowFullChangelog(true);
  }

  function closeFullChangelog() {
    mdAbort.current?.abort();
    setShowFullChangelog(false);
    setEntryMarkdown(null);
    setSelectedEntry(null);
  }

  function selectEntry(entry: ChangelogEntry) {
    setSelectedEntry(entry);
    fetchEntryMarkdown(entry);
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

        setTimeout(() => {
          if (!cancelled) setIsVisible(true);
        }, 1500);
      })
      .catch(() => {
        /* non-critical */
      });

    return () => {
      cancelled = true;
      clearTimers();
      mdAbort.current?.abort();
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
