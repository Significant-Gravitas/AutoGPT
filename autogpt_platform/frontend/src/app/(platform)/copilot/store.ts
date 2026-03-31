import { Key, storage } from "@/services/storage/local-storage";
import { create } from "zustand";

export interface DeleteTarget {
  id: string;
  title: string | null | undefined;
}

export interface ArtifactRef {
  id: string;
  title: string;
  mimeType: string | null;
  sourceUrl: string;
  origin: "agent" | "user-upload";
  sizeBytes?: number;
}

interface ArtifactPanelState {
  isOpen: boolean;
  isMinimized: boolean;
  isMaximized: boolean;
  width: number;
  activeArtifact: ArtifactRef | null;
  history: ArtifactRef[];
}

const DEFAULT_PANEL_WIDTH = 600;

function getPersistedWidth(): number {
  const saved = storage.get(Key.COPILOT_ARTIFACT_PANEL_WIDTH);
  if (saved) {
    const parsed = parseInt(saved, 10);
    if (!isNaN(parsed) && parsed >= 320) return parsed;
  }
  return DEFAULT_PANEL_WIDTH;
}

interface CopilotUIState {
  /** Prompt extracted from URL hash (e.g. /copilot#prompt=...) for input prefill. */
  initialPrompt: string | null;
  setInitialPrompt: (prompt: string | null) => void;

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

  // Artifact panel
  artifactPanel: ArtifactPanelState;
  openArtifact: (ref: ArtifactRef) => void;
  closeArtifactPanel: () => void;
  minimizeArtifactPanel: () => void;
  maximizeArtifactPanel: () => void;
  restoreArtifactPanel: () => void;
  setArtifactPanelWidth: (width: number) => void;
  goBackArtifact: () => void;

  clearCopilotLocalData: () => void;
}

export const useCopilotUIStore = create<CopilotUIState>((set) => ({
  initialPrompt: null,
  setInitialPrompt: (prompt) => set({ initialPrompt: prompt }),

  sessionToDelete: null,
  setSessionToDelete: (target) => set({ sessionToDelete: target }),

  isDrawerOpen: false,
  setDrawerOpen: (open) => set({ isDrawerOpen: open }),

  completedSessionIDs: new Set<string>(),
  addCompletedSession: (id) =>
    set((state) => {
      const next = new Set(state.completedSessionIDs);
      next.add(id);
      return { completedSessionIDs: next };
    }),
  clearCompletedSession: (id) =>
    set((state) => {
      const next = new Set(state.completedSessionIDs);
      next.delete(id);
      return { completedSessionIDs: next };
    }),
  clearAllCompletedSessions: () =>
    set({ completedSessionIDs: new Set<string>() }),

  isNotificationsEnabled:
    storage.get(Key.COPILOT_NOTIFICATIONS_ENABLED) === "true" &&
    typeof Notification !== "undefined" &&
    Notification.permission === "granted",
  setNotificationsEnabled: (enabled) => {
    storage.set(Key.COPILOT_NOTIFICATIONS_ENABLED, String(enabled));
    set({ isNotificationsEnabled: enabled });
  },

  isSoundEnabled: storage.get(Key.COPILOT_SOUND_ENABLED) !== "false",
  toggleSound: () =>
    set((state) => {
      const next = !state.isSoundEnabled;
      storage.set(Key.COPILOT_SOUND_ENABLED, String(next));
      return { isSoundEnabled: next };
    }),

  showNotificationDialog: false,
  setShowNotificationDialog: (show) => set({ showNotificationDialog: show }),

  // Artifact panel
  artifactPanel: {
    isOpen: false,
    isMinimized: false,
    isMaximized: false,
    width: getPersistedWidth(),
    activeArtifact: null,
    history: [],
  },
  openArtifact: (ref) =>
    set((state) => {
      const history =
        state.artifactPanel.activeArtifact &&
        state.artifactPanel.activeArtifact.id !== ref.id
          ? [...state.artifactPanel.history, state.artifactPanel.activeArtifact]
          : state.artifactPanel.history;
      return {
        artifactPanel: {
          ...state.artifactPanel,
          isOpen: true,
          isMinimized: false,
          activeArtifact: ref,
          history,
        },
      };
    }),
  closeArtifactPanel: () =>
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isOpen: false,
        isMinimized: false,
        // Keep activeArtifact so exit animations can reference it
        history: [],
      },
    })),
  minimizeArtifactPanel: () =>
    set((state) => ({
      artifactPanel: { ...state.artifactPanel, isMinimized: true },
    })),
  maximizeArtifactPanel: () =>
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isMaximized: true,
        isMinimized: false,
      },
    })),
  restoreArtifactPanel: () =>
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isMaximized: false,
        isMinimized: false,
      },
    })),
  setArtifactPanelWidth: (width) => {
    storage.set(Key.COPILOT_ARTIFACT_PANEL_WIDTH, String(width));
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        width,
        isMaximized: false,
      },
    }));
  },
  goBackArtifact: () =>
    set((state) => {
      const { history } = state.artifactPanel;
      if (history.length === 0) return state;
      const previous = history[history.length - 1];
      return {
        artifactPanel: {
          ...state.artifactPanel,
          activeArtifact: previous,
          history: history.slice(0, -1),
        },
      };
    }),

  clearCopilotLocalData: () => {
    storage.clean(Key.COPILOT_NOTIFICATIONS_ENABLED);
    storage.clean(Key.COPILOT_SOUND_ENABLED);
    storage.clean(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED);
    storage.clean(Key.COPILOT_NOTIFICATION_DIALOG_DISMISSED);
    storage.clean(Key.COPILOT_ARTIFACT_PANEL_WIDTH);
    set({
      completedSessionIDs: new Set<string>(),
      isNotificationsEnabled: false,
      isSoundEnabled: true,
      artifactPanel: {
        isOpen: false,
        isMinimized: false,
        isMaximized: false,
        width: DEFAULT_PANEL_WIDTH,
        activeArtifact: null,
        history: [],
      },
    });
    document.title = "AutoGPT";
  },
}));
