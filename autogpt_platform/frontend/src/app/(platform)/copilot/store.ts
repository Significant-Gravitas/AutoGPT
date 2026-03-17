import { Key, storage } from "@/services/storage/local-storage";
import { create } from "zustand";

export interface DeleteTarget {
  id: string;
  title: string | null | undefined;
}

const isClient = typeof window !== "undefined";

function loadCompletedSessions(): Set<string> {
  if (!isClient) return new Set();
  const raw = storage.get(Key.COPILOT_COMPLETED_SESSIONS);
  if (!raw) return new Set();
  try {
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? new Set<string>(parsed.filter((v) => typeof v === "string"))
      : new Set();
  } catch {
    return new Set();
  }
}

function persistCompletedSessions(ids: Set<string>) {
  if (!isClient) return;
  try {
    if (ids.size === 0) {
      storage.clean(Key.COPILOT_COMPLETED_SESSIONS);
    } else {
      storage.set(Key.COPILOT_COMPLETED_SESSIONS, JSON.stringify([...ids]));
    }
  } catch {
    // Keep in-memory state authoritative if persistence is unavailable
  }
}

interface CopilotUIState {
  sessionToDelete: DeleteTarget | null;
  setSessionToDelete: (target: DeleteTarget | null) => void;

  isDrawerOpen: boolean;
  setDrawerOpen: (open: boolean) => void;

  completedSessionIDs: Set<string>;
  addCompletedSession: (id: string) => void;
  clearCompletedSession: (id: string) => void;
  clearAllCompletedSessions: () => void;

  isNotificationsEnabled: boolean;
  setNotificationsEnabled: (enabled: boolean) => void;

  isSoundEnabled: boolean;
  toggleSound: () => void;

  showNotificationDialog: boolean;
  setShowNotificationDialog: (show: boolean) => void;

  clearCopilotLocalData: () => void;
}

export const useCopilotUIStore = create<CopilotUIState>((set) => ({
  sessionToDelete: null,
  setSessionToDelete: (target) => set({ sessionToDelete: target }),

  isDrawerOpen: false,
  setDrawerOpen: (open) => set({ isDrawerOpen: open }),

  completedSessionIDs: loadCompletedSessions(),
  addCompletedSession: (id) =>
    set((state) => {
      const next = new Set(state.completedSessionIDs);
      next.add(id);
      persistCompletedSessions(next);
      return { completedSessionIDs: next };
    }),
  clearCompletedSession: (id) =>
    set((state) => {
      const next = new Set(state.completedSessionIDs);
      next.delete(id);
      persistCompletedSessions(next);
      return { completedSessionIDs: next };
    }),
  clearAllCompletedSessions: () => {
    persistCompletedSessions(new Set());
    set({ completedSessionIDs: new Set<string>() });
  },

  isNotificationsEnabled:
    isClient &&
    storage.get(Key.COPILOT_NOTIFICATIONS_ENABLED) === "true" &&
    typeof Notification !== "undefined" &&
    Notification.permission === "granted",
  setNotificationsEnabled: (enabled) => {
    storage.set(Key.COPILOT_NOTIFICATIONS_ENABLED, String(enabled));
    set({ isNotificationsEnabled: enabled });
  },

  isSoundEnabled:
    !isClient || storage.get(Key.COPILOT_SOUND_ENABLED) !== "false",
  toggleSound: () =>
    set((state) => {
      const next = !state.isSoundEnabled;
      storage.set(Key.COPILOT_SOUND_ENABLED, String(next));
      return { isSoundEnabled: next };
    }),

  showNotificationDialog: false,
  setShowNotificationDialog: (show) => set({ showNotificationDialog: show }),

  clearCopilotLocalData: () => {
    storage.clean(Key.COPILOT_NOTIFICATIONS_ENABLED);
    storage.clean(Key.COPILOT_SOUND_ENABLED);
    storage.clean(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED);
    storage.clean(Key.COPILOT_NOTIFICATION_DIALOG_DISMISSED);
    storage.clean(Key.COPILOT_COMPLETED_SESSIONS);
    set({
      completedSessionIDs: new Set<string>(),
      isNotificationsEnabled: false,
      isSoundEnabled: true,
    });
    document.title = "AutoGPT";
  },
}));
