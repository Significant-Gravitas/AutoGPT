import { create } from "zustand";

const DELIMITTER = "_#_";

interface HandleStore {
  fromRjsfId: (id: string) => string; // Convert RJSF id to handle id
  composeKey: (parts: string[]) => string; // Compose a nested key for handles; consistent with legacy builder
  normalizeKey: (key: string) => string;
}

export const useHandleStore = create<HandleStore>((set, get) => ({
  fromRjsfId: (id: string) => {
    if (!id) return "";
    const parts = id.split("_");
    const filtered = parts.filter(
      (p) => p !== "root" && p !== "properties" && p.length > 0,
    );
    return filtered.at(-1) || "";
  },

  composeKey: (parts: string[]) => {
    const cleaned = parts.filter(Boolean).map((p) => get().normalizeKey(p));
    return cleaned.join(DELIMITTER);
  },

  normalizeKey: (key: string) => {
    return key
      .trim()
      .replace(/\s+/g, "_")
      .replace(/[^a-zA-Z0-9_\-]/g, "_")
      .toLowerCase();
  },
}));
