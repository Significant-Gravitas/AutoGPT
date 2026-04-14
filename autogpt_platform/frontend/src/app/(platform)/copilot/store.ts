import { Key, storage } from "@/services/storage/local-storage";
import { create } from "zustand";
import { clearContentCache } from "./components/ArtifactPanel/components/useArtifactContent";
import { classifyArtifact } from "./components/ArtifactPanel/helpers";
import { ORIGINAL_TITLE, parseSessionIDs } from "./helpers";

export interface DeleteTarget {
  id: string;
  title: string | null | undefined;
}

/**
 * A single workspace artifact surfaced in the copilot chat.
 *
 * Rendered by `ArtifactCard` (inline) and `ArtifactPanel` (preview pane).
 * Typically extracted from `workspace://<id>` URIs in assistant text parts
 * or from `FileUIPart` attachments; see `getMessageArtifacts` in
 * `ChatMessagesContainer/helpers.ts`.
 */
export interface ArtifactRef {
  /** Workspace file ID (matches the backend `WorkspaceFile.id`). */
  id: string;
  /** Human-visible filename, used as both title and download filename. */
  title: string;
  /** MIME type if known (from backend metadata or `workspace://id#mime`). */
  mimeType: string | null;
  /**
   * Fully-qualified URL the preview/download code will fetch from. Today
   * this is always the same-origin proxy path
   * `/api/proxy/api/workspace/files/{id}/download`.
   */
  sourceUrl: string;
  /**
   * Who produced the artifact — drives the origin badge color in
   * `ArtifactPanelHeader`. Derived from the emitting message's role.
   */
  origin: "agent" | "user-upload";
  /** Size in bytes if known — used by `classifyArtifact` for size gating. */
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

export const DEFAULT_PANEL_WIDTH = 600;

/** Autopilot response mode. */
export type CopilotMode = "extended_thinking" | "fast";

const isClient = typeof window !== "undefined";

function getPersistedWidth(): number {
  if (!isClient) return DEFAULT_PANEL_WIDTH;
  const saved = storage.get(Key.COPILOT_ARTIFACT_PANEL_WIDTH);
  if (saved) {
    const parsed = parseInt(saved, 10);
    // Match the drag-handle clamp so a stale/corrupt value can't open the
    // panel wider than 85% of the viewport.
    const maxWidth = window.innerWidth * 0.85;
    if (!isNaN(parsed) && parsed >= 320) {
      return Math.min(parsed, maxWidth);
    }
  }
  return DEFAULT_PANEL_WIDTH;
}

let panelWidthPersistTimer: ReturnType<typeof setTimeout> | null = null;
function schedulePanelWidthPersist(width: number) {
  if (!isClient) return;
  if (panelWidthPersistTimer) clearTimeout(panelWidthPersistTimer);
  panelWidthPersistTimer = setTimeout(() => {
    storage.set(Key.COPILOT_ARTIFACT_PANEL_WIDTH, String(width));
    panelWidthPersistTimer = null;
  }, 200);
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

function isPreviewableArtifact(ref: ArtifactRef): boolean {
  return classifyArtifact(ref.mimeType, ref.title, ref.sizeBytes).openable;
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
  resetArtifactPanel: () => void;
  minimizeArtifactPanel: () => void;
  maximizeArtifactPanel: () => void;
  restoreArtifactPanel: () => void;
  setArtifactPanelWidth: (width: number) => void;
  goBackArtifact: () => void;

  /** Autopilot mode: 'extended_thinking' (default) or 'fast'. */
  copilotMode: CopilotMode;
  setCopilotMode: (mode: CopilotMode) => void;

  /** Developer dry-run mode: sessions created with dry_run=true. */
  isDryRun: boolean;
  setIsDryRun: (enabled: boolean) => void;

  clearCopilotLocalData: () => void;
}

export const useCopilotUIStore = create<CopilotUIState>((set) => ({
  initialPrompt: null,
  setInitialPrompt: (prompt) => set({ initialPrompt: prompt }),

  sessionToDelete: null,
  setSessionToDelete: (target) => set({ sessionToDelete: target }),

  isDrawerOpen: false,
  setDrawerOpen: (open) => set({ isDrawerOpen: open }),

  completedSessionIDs: isClient
    ? parseSessionIDs(storage.get(Key.COPILOT_COMPLETED_SESSIONS))
    : new Set(),
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
      if (!isPreviewableArtifact(ref)) return state;

      const { activeArtifact, history: prevHistory } = state.artifactPanel;
      const topOfHistory = prevHistory[prevHistory.length - 1];
      const isReturningToTop = topOfHistory?.id === ref.id;
      const shouldPushHistory =
        state.artifactPanel.isOpen &&
        activeArtifact != null &&
        activeArtifact.id !== ref.id;
      const MAX_HISTORY = 25;
      const history = isReturningToTop
        ? prevHistory.slice(0, -1)
        : shouldPushHistory
          ? [...prevHistory, activeArtifact!].slice(-MAX_HISTORY)
          : prevHistory;
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
        history: [],
      },
    })),
  resetArtifactPanel: () =>
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isOpen: false,
        isMinimized: false,
        isMaximized: false,
        activeArtifact: null,
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
    schedulePanelWidthPersist(width);
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

  copilotMode: "extended_thinking",
  setCopilotMode: (mode) => {
    set({ copilotMode: mode });
  },

  isDryRun: isClient && storage.get(Key.COPILOT_DRY_RUN) === "true",
  setIsDryRun: (enabled) => {
    if (enabled) {
      storage.set(Key.COPILOT_DRY_RUN, "true");
    } else {
      storage.clean(Key.COPILOT_DRY_RUN);
    }
    set({ isDryRun: enabled });
  },

  clearCopilotLocalData: () => {
    clearContentCache();
    storage.clean(Key.COPILOT_NOTIFICATIONS_ENABLED);
    storage.clean(Key.COPILOT_SOUND_ENABLED);
    storage.clean(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED);
    storage.clean(Key.COPILOT_NOTIFICATION_DIALOG_DISMISSED);
    storage.clean(Key.COPILOT_ARTIFACT_PANEL_WIDTH);
    storage.clean(Key.COPILOT_COMPLETED_SESSIONS);
    storage.clean(Key.COPILOT_DRY_RUN);
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
      copilotMode: "extended_thinking",
      isDryRun: false,
    });
    if (isClient) {
      document.title = ORIGINAL_TITLE;
    }
  },
}));
