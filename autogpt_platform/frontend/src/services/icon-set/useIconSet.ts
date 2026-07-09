"use client";

import { create } from "zustand";

export type IconSet = "pika" | "phosphor";

const STORAGE_KEY = "agpt-icon-set";

function readStored(): IconSet | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  return raw === "pika" || raw === "phosphor" ? raw : null;
}

interface IconSetState {
  /** User-chosen icon set, or `null` to follow the `PIKA_ICONS` flag. */
  iconSet: IconSet | null;
  setIconSet: (value: IconSet) => void;
}

export const useIconSetStore = create<IconSetState>((set) => ({
  iconSet: readStored(),
  setIconSet: (value) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, value);
    }
    set({ iconSet: value });
  },
}));
