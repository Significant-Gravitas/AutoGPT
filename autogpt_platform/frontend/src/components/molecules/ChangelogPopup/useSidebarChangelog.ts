"use client";

import { useEffect, useRef, useState } from "react";
import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "./changelog-constants";
import {
  ChangelogEntry,
  cleanEntryMarkdown,
  parseChangelogIndex,
} from "./helpers";

// Passive changelog surface for the sidebar footer: no auto-popup or dismiss
// timers, just an "unseen" indicator and the full modal on demand. The
// floating ChangelogPopup owns the proactive toast on the classic layout.
export function useSidebarChangelog() {
  const [entries, setEntries] = useState<ChangelogEntry[]>([]);
  const [hasUnseen, setHasUnseen] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedEntry, setSelectedEntry] = useState<ChangelogEntry | null>(
    null,
  );
  const [entryMarkdown, setEntryMarkdown] = useState<string | null>(null);
  const [isLoadingMarkdown, setIsLoadingMarkdown] = useState(false);

  const mdAbort = useRef<AbortController | null>(null);

  function markLatestSeen(slug: string) {
    try {
      localStorage.setItem(STORAGE_KEY, slug);
    } catch {
      /* noop */
    }
    setHasUnseen(false);
  }

  function loadEntryMarkdown(entry: ChangelogEntry) {
    mdAbort.current?.abort();
    const controller = new AbortController();
    mdAbort.current = controller;

    setIsLoadingMarkdown(true);
    setEntryMarkdown(null);

    fetch(entry.mdUrl, { signal: controller.signal })
      .then((res) => (res.ok ? res.text() : ""))
      .then((md) => {
        if (controller.signal.aborted) return;
        setEntryMarkdown(cleanEntryMarkdown(md));
      })
      .catch(() => {
        /* abort or network error — non-critical */
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoadingMarkdown(false);
      });
  }

  function open() {
    const latest = entries[0];
    setIsOpen(true);
    if (latest) {
      setSelectedEntry(latest);
      loadEntryMarkdown(latest);
      markLatestSeen(latest.slug);
    }
  }

  function close() {
    mdAbort.current?.abort();
    setIsOpen(false);
    setSelectedEntry(null);
    setEntryMarkdown(null);
  }

  function selectEntry(entry: ChangelogEntry) {
    setSelectedEntry(entry);
    loadEntryMarkdown(entry);
  }

  useEffect(() => {
    let cancelled = false;

    fetch(CHANGELOG_INDEX_MD_URL)
      .then((res) => (res.ok ? res.text() : ""))
      .then((md) => {
        if (cancelled || !md) return;

        const parsed = parseChangelogIndex(md);
        if (parsed.length === 0) return;

        setEntries(parsed);

        try {
          const lastSeen = localStorage.getItem(STORAGE_KEY);
          setHasUnseen(lastSeen !== parsed[0].slug);
        } catch {
          setHasUnseen(true);
        }
      })
      .catch(() => {
        /* non-critical */
      });

    return () => {
      cancelled = true;
      mdAbort.current?.abort();
    };
  }, []);

  return {
    entries,
    hasUnseen,
    isOpen,
    open,
    close,
    selectedEntry,
    selectEntry,
    entryMarkdown,
    isLoadingMarkdown,
  };
}
