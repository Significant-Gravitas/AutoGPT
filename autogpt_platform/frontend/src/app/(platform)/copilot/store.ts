import { Key, storage } from "@/services/storage/local-storage";
import { create } from "zustand";
import { clearContentCache } from "./components/ArtifactPanel/components/useArtifactContent";
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
  activeTab: ContextPanelTab;
}

export const DEFAULT_PANEL_WIDTH = 272; // 17rem
export const MAX_PANEL_WIDTH = 280; // 17.5rem

/** Autopilot response mode. */
export type CopilotMode = "extended_thinking" | "fast";

/** Per-request model tier. 'standard' = current default; 'advanced' = highest-capability. */
export type CopilotLlmModel = "standard" | "advanced";

/** Context panel tab. */
export type ContextPanelTab = "progress" | "files";

const isClient = typeof window !== "undefined";

function getPersistedWidth(): number {
  if (!isClient) return DEFAULT_PANEL_WIDTH;
  const saved = storage.get(Key.COPILOT_ARTIFACT_PANEL_WIDTH);
  if (saved) {
    const parsed = parseInt(saved, 10);
    // Clamp stale persisted values to the current MAX so a previously larger
    // panel doesn't reopen above the new cap.
    if (!isNaN(parsed) && parsed >= 240) {
      return Math.min(parsed, MAX_PANEL_WIDTH);
    }
  }
  return DEFAULT_PANEL_WIDTH;
}

function getPersistedOpen(): boolean {
  if (!isClient) return false;
  return storage.get(Key.COPILOT_CONTEXT_PANEL_OPEN) === "true";
}

function getPersistedTab(): ContextPanelTab {
  if (!isClient) return "files";
  const saved = storage.get(Key.COPILOT_CONTEXT_PANEL_TAB);
  return saved === "progress" ? saved : "files";
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

interface CopilotUIState {
  /** Prompt extracted from URL hash (e.g. /copilot#prompt=...) for input prefill. */
  initialPrompt: string | null;
  setInitialPrompt: (prompt: string | null) => void;

  sessionToDelete: DeleteTarget | null;
  setSessionToDelete: (target: DeleteTarget | null) => void;

  isDrawerOpen: boolean;
  setDrawerOpen: (open: boolean) => void;

  isSearchOpen: boolean;
  setSearchOpen: (open: boolean) => void;

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
  clearArtifactPreview: () => void;
  resetArtifactPanel: () => void;
  minimizeArtifactPanel: () => void;
  maximizeArtifactPanel: () => void;
  restoreArtifactPanel: () => void;
  setArtifactPanelWidth: (width: number) => void;
  goBackArtifact: () => void;
  setActiveTab: (tab: ContextPanelTab) => void;
  toggleContextPanel: () => void;
  openContextPanelForFiles: () => void;
  openContextPanelForProgress: () => void;

  // Card-based auto-open: ArtifactCard registers itself on mount, the store
  // decides whether to auto-open. Much simpler than message-scanning.
  registerArtifactForAutoOpen: (ref: ArtifactRef) => void;
  setAutoOpenReady: () => void;
  markUserClosedForAutoOpen: () => void;
  resetAutoOpenState: () => void;

  /** Autopilot mode: 'extended_thinking' (default) or 'fast'. */
  copilotChatMode: CopilotMode;
  setCopilotChatMode: (mode: CopilotMode) => void;

  /** Model tier: 'standard' (default) or 'advanced' (highest-capability). */
  copilotLlmModel: CopilotLlmModel;
  setCopilotLlmModel: (model: CopilotLlmModel) => void;

  /** Developer dry-run mode: sessions created with dry_run=true. */
  isDryRun: boolean;
  setIsDryRun: (enabled: boolean) => void;

  clearCopilotLocalData: () => void;
}

// ── Card-based auto-open tracking ───────────────────────────────────
// Module-level state — not in Zustand to avoid unnecessary re-renders.
// ArtifactCard calls registerArtifactForAutoOpen on mount; the store
// decides whether to auto-open based on these flags.
const _autoOpenKnownIds = new Set<string>();
let _autoOpenReady = false;
let _autoOpenUserClosed = false;

export const useCopilotUIStore = create<CopilotUIState>((set, get) => ({
  initialPrompt: null,
  setInitialPrompt: (prompt) => set({ initialPrompt: prompt }),

  sessionToDelete: null,
  setSessionToDelete: (target) => set({ sessionToDelete: target }),

  isDrawerOpen: false,
  setDrawerOpen: (open) => set({ isDrawerOpen: open }),

  isSearchOpen: false,
  setSearchOpen: (open) => set({ isSearchOpen: open }),

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
    isOpen: getPersistedOpen(),
    isMinimized: false,
    isMaximized: false,
    width: getPersistedWidth(),
    activeArtifact: null,
    history: [],
    activeTab: getPersistedTab(),
  },
  openArtifact: (ref) =>
    set((state) => {
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
    set((state) => {
      if (isClient) storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "false");
      return {
        artifactPanel: {
          ...state.artifactPanel,
          isOpen: false,
          isMinimized: false,
          history: [],
        },
      };
    }),
  clearArtifactPreview: () =>
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        activeArtifact: null,
        history: [],
        isMinimized: false,
        isMaximized: false,
      },
    })),
  resetArtifactPanel: () =>
    set((state) => ({
      // Clear preview state only — leave `isOpen` alone since it's shared
      // with ContextPanel, which would otherwise collapse on session
      // switches (resetArtifactPanel runs in useAutoOpenArtifacts).
      artifactPanel: {
        ...state.artifactPanel,
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
  setActiveTab: (tab) =>
    set((state) => {
      if (isClient) storage.set(Key.COPILOT_CONTEXT_PANEL_TAB, tab);
      return {
        // Selecting a tab returns to the tabs view (drops any open preview).
        artifactPanel: {
          ...state.artifactPanel,
          activeTab: tab,
          activeArtifact: null,
          history: [],
          isMinimized: false,
        },
      };
    }),
  toggleContextPanel: () =>
    set((state) => {
      const nextOpen = !state.artifactPanel.isOpen;
      if (isClient)
        storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, String(nextOpen));
      // Persist the new open state. Opening always clears any previous
      // preview so the toggle lands on the tabs view (closeArtifactPanel /
      // the preview's X button leave activeArtifact set); closing counts as
      // an explicit close for files auto-open.
      if (!nextOpen) _autoOpenUserClosed = true;
      return {
        artifactPanel: {
          ...state.artifactPanel,
          isOpen: nextOpen,
          isMinimized: false,
          activeArtifact: nextOpen ? null : state.artifactPanel.activeArtifact,
          history: nextOpen ? [] : state.artifactPanel.history,
        },
      };
    }),
  openContextPanelForFiles: () => {
    if (_autoOpenUserClosed) return;
    if (get().artifactPanel.isOpen) return;
    if (isClient) {
      storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "true");
      storage.set(Key.COPILOT_CONTEXT_PANEL_TAB, "files");
    }
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isOpen: true,
        isMinimized: false,
        activeTab: "files",
        activeArtifact: null,
        history: [],
      },
    }));
  },
  openContextPanelForProgress: () => {
    if (isClient) {
      storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "true");
      storage.set(Key.COPILOT_CONTEXT_PANEL_TAB, "progress");
    }
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isOpen: true,
        isMinimized: false,
        activeTab: "progress",
        activeArtifact: null,
        history: [],
      },
    }));
  },

  // ── Card-based auto-open actions ─────────────────────────────────
  registerArtifactForAutoOpen: (ref) => {
    // Auto-open is disabled — the drawer only opens on explicit click.
    // We still track the ID so we can upgrade activeArtifact metadata when
    // a richer ref (e.g. file-part with real MIME) arrives for the same id.
    if (_autoOpenKnownIds.has(ref.id)) {
      const active = get().artifactPanel.activeArtifact;
      if (active?.id === ref.id && !active.mimeType && ref.mimeType) {
        set((state) => ({
          artifactPanel: { ...state.artifactPanel, activeArtifact: ref },
        }));
      }
      return;
    }
    _autoOpenKnownIds.add(ref.id);
  },
  setAutoOpenReady: () => {
    _autoOpenReady = true;
  },
  markUserClosedForAutoOpen: () => {
    _autoOpenUserClosed = true;
  },
  resetAutoOpenState: () => {
    _autoOpenKnownIds.clear();
    _autoOpenReady = false;
    _autoOpenUserClosed = false;
  },

  copilotChatMode: (() => {
    const saved = isClient ? storage.get(Key.COPILOT_MODE) : null;
    return saved === "fast" ? "fast" : "extended_thinking";
  })(),
  setCopilotChatMode: (mode) => {
    storage.set(Key.COPILOT_MODE, mode);
    set({ copilotChatMode: mode });
  },

  copilotLlmModel: (() => {
    const saved = isClient ? storage.get(Key.COPILOT_MODEL) : null;
    return saved === "advanced" ? "advanced" : "standard";
  })(),
  setCopilotLlmModel: (model) => {
    storage.set(Key.COPILOT_MODEL, model);
    set({ copilotLlmModel: model });
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
    _autoOpenKnownIds.clear();
    _autoOpenReady = false;
    _autoOpenUserClosed = false;
    storage.clean(Key.COPILOT_NOTIFICATIONS_ENABLED);
    storage.clean(Key.COPILOT_SOUND_ENABLED);
    storage.clean(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED);
    storage.clean(Key.COPILOT_NOTIFICATION_DIALOG_DISMISSED);
    storage.clean(Key.COPILOT_ARTIFACT_PANEL_WIDTH);
    storage.clean(Key.COPILOT_CONTEXT_PANEL_OPEN);
    storage.clean(Key.COPILOT_CONTEXT_PANEL_TAB);
    storage.clean(Key.COPILOT_COMPLETED_SESSIONS);
    storage.clean(Key.COPILOT_DRY_RUN);
    storage.clean(Key.COPILOT_MODE);
    storage.clean(Key.COPILOT_MODEL);
    set({
      completedSessionIDs: new Set<string>(),
      isSearchOpen: false,
      isNotificationsEnabled: false,
      isSoundEnabled: true,
      artifactPanel: {
        isOpen: false,
        isMinimized: false,
        isMaximized: false,
        width: DEFAULT_PANEL_WIDTH,
        activeArtifact: null,
        history: [],
        activeTab: "files",
      },
      copilotChatMode: "extended_thinking",
      copilotLlmModel: "standard",
      isDryRun: false,
    });
    if (isClient) {
      document.title = ORIGINAL_TITLE;
    }
  },
}));
