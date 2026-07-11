"use client";

import { useEffect, useState } from "react";
import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "./changelog-constants";
import { ChangelogEntry, parseChangelogIndex } from "./helpers";

// Passive changelog surface for the sidebar footer: an "unseen" indicator and
// a modal listing the releases (each links out to the docs). The floating
// ChangelogPopup owns the proactive toast on the classic layout.
export function useSidebarChangelog() {
  const [entries, setEntries] = useState<ChangelogEntry[]>([]);
  const [hasUnseen, setHasUnseen] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  function markLatestSeen(slug: string) {
    try {
      localStorage.setItem(STORAGE_KEY, slug);
    } catch {
      /* noop */
    }
    setHasUnseen(false);
  }

  function open() {
    const latest = entries[0];
    // Guard against opening an empty modal before the index has loaded.
    if (!latest) return;
    setIsOpen(true);
    markLatestSeen(latest.slug);
  }

  function close() {
    setIsOpen(false);
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
    };
  }, []);

  return {
    entries,
    hasUnseen,
    isOpen,
    open,
    close,
  };
}
