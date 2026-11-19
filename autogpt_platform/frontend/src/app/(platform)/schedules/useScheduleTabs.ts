"use client";

import { useState } from "react";

const NARROW = 380;
const WIDE = 520;

/** The schedules open in the side panel, as tabs like the chat's work area. */
export function useScheduleTabs() {
  const [tabs, setTabs] = useState<string[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [isWide, setIsWide] = useState(false);

  function open(id: string) {
    setTabs((current) => (current.includes(id) ? current : [...current, id]));
    setActive(id);
    setIsOpen(true);
  }

  function close(id: string) {
    const rest = tabs.filter((tab) => tab !== id);
    setTabs(rest);
    if (active === id) setActive(rest[rest.length - 1] ?? null);
    if (!rest.length) setIsOpen(false);
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
  };
}
