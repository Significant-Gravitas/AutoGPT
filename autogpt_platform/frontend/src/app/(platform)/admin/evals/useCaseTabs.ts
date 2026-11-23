"use client";

import { useState } from "react";

const NARROW = 400;
const WIDE = 560;

/** The cases open in the side panel, as tabs like the chat's work area. */
export function useCaseTabs() {
  const [tabs, setTabs] = useState<string[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [isWide, setIsWide] = useState(false);

  function open(caseId: string) {
    setTabs((current) =>
      current.includes(caseId) ? current : [...current, caseId],
    );
    setActive(caseId);
    setIsOpen(true);
  }

  function close(caseId: string) {
    const rest = tabs.filter((id) => id !== caseId);
    setTabs(rest);
    if (active === caseId) setActive(rest[rest.length - 1] ?? null);
    if (!rest.length) setIsOpen(false);
  }

  function reset() {
    setTabs([]);
    setActive(null);
    setIsOpen(false);
  }

  return {
    tabs,
    active,
    isOpen: isOpen && tabs.length > 0,
    width: isWide ? WIDE : NARROW,
    isWide,
    open,
    close,
    select: setActive,
    fold: () => setIsOpen(false),
    toggleWidth: () => setIsWide((wide) => !wide),
    reset,
  };
}
