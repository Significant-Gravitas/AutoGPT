"use client";

import { useState } from "react";

const NARROW = 400;
const WIDE = 560;

/** The block runs open beside the run, as tabs like the chat's work area. */
export function useStepTabs() {
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
    reset,
    select: setActive,
    fold: () => setIsOpen(false),
    toggleWidth: () => setIsWide((wide) => !wide),
  };
}
