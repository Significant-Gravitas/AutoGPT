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
  activeArtifact: ArtifactRef | null;
  history: ArtifactRef[];
  activeTab: ContextPanelTab;
  /** Preview that was showing when the panel last closed — the sidebar
   *  toggle restores it on reopen instead of landing on the tabs view. */
  lastArtifact: ArtifactRef | null;
}

export const DEFAULT_PANEL_WIDTH = 432; // context panel default (352 + 80)
export const DEFAULT_ARTIFACT_PANEL_WIDTH = 640;
export const MIN_CONTEXT_PANEL_WIDTH = 280;
export const MAX_CONTEXT_PANEL_WIDTH = 600;
export const MIN_ARTIFACT_PANEL_WIDTH = 400;
/** Space kept for the chat + rail when sizing a side panel (drag clamp and viewport clamp). */
export const PANEL_RESERVED_WIDTH = 440;

/** Autopilot response mode. */
export type CopilotMode = "extended_thinking" | "fast";

/** Per-request model tier. 'standard' = current default; 'advanced' = highest-capability. */
export type CopilotLlmModel = "standard" | "advanced";

export type CopilotLlmAuthSelection =
  | { authProvider: "platform"; credentialId: null }
  | { authProvider: "codex"; credentialId: string };

/** Context panel tab: "files" is the inline workspace-files card, "artifacts"
 *  the docked artifacts library. */
export type ContextPanelTab = "files" | "artifacts";

const isClient = typeof window !== "undefined";

function getPersistedOpen(): boolean {
  if (!isClient) return false;
  return storage.get(Key.COPILOT_CONTEXT_PANEL_OPEN) === "true";
}

function getPersistedTab(): ContextPanelTab {
  if (!isClient) return "files";
  const saved = storage.get(Key.COPILOT_CONTEXT_PANEL_TAB);
  // Anything else (including a "progress" tab persisted by the retired
  // sidebar) falls back to the files card.
  return saved === "artifacts" ? saved : "files";
}

function clampWidth(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function getPersistedContextWidth(): number {
  if (!isClient) return DEFAULT_PANEL_WIDTH;
  const raw = Number(storage.get(Key.COPILOT_CONTEXT_PANEL_WIDTH));
  return Number.isFinite(raw) && raw > 0
    ? clampWidth(raw, MIN_CONTEXT_PANEL_WIDTH, MAX_CONTEXT_PANEL_WIDTH)
    : DEFAULT_PANEL_WIDTH;
}

function getPersistedArtifactWidth(): number {
  if (!isClient) return DEFAULT_ARTIFACT_PANEL_WIDTH;
  const raw = Number(storage.get(Key.COPILOT_ARTIFACT_PANEL_WIDTH));
  return Number.isFinite(raw) && raw >= MIN_ARTIFACT_PANEL_WIDTH
    ? raw
    : DEFAULT_ARTIFACT_PANEL_WIDTH;
}

const widthPersistTimers: Record<string, ReturnType<typeof setTimeout>> = {};
function scheduleWidthPersist(key: Key, value: number) {
  if (!isClient) return;
  const existing = widthPersistTimers[key];
  if (existing) clearTimeout(existing);
  widthPersistTimers[key] = setTimeout(() => {
    storage.set(key, String(Math.round(value)));
    delete widthPersistTimers[key];
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

  /**
   * Bumped every time the composer actually sends a message. Chain action
   * cards draft into the input rather than sending themselves, so this is
   * how they learn their drafted message went out.
   */
  sentMessageCount: number;
  notifyMessageSent: () => void;

  /**
   * Expert ids whose latest thread was already adopted via a
   * /copilot?expertId= deep link this page load. Lives here — not in
   * useChatSession refs — because the chat host remounts on every sessionId
   * change (CopilotPage keys the subtree), wiping hook refs; a wiped latch
   * made "New Chat" bounce straight back into the adopted thread.
   */
  adoptedExpertThreads: Set<string>;
  markExpertThreadAdopted: (expertId: string) => void;

  contextPanelWidth: number;
  artifactPanelWidth: number;
  setContextPanelWidth: (width: number) => void;
  setArtifactPanelWidth: (width: number) => void;

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
  // `persist: false` skips the localStorage write — used by the public
  // /tour demo so a scripted open/close never leaks into the panel state
  // the real copilot restores on load.
  openArtifact: (ref: ArtifactRef, opts?: { persist?: boolean }) => void;
  closeArtifactPanel: (opts?: { persist?: boolean }) => void;
  clearArtifactPreview: () => void;
  resetArtifactPanel: () => void;
  goBackArtifact: () => void;
  toggleContextPanel: () => void;
  /** Opens the panel on `tab`, or closes it if that tab is already showing. */
  toggleContextPanelTab: (tab: ContextPanelTab) => void;
  /** Forget the remembered preview — called on session entry so a new chat
   *  can never restore the previous chat's artifact. */
  clearLastArtifact: () => void;
  openContextPanelForFiles: () => void;
  autoOpenArtifact: (ref: ArtifactRef) => void;
  showFilesTab: () => void;

  // Card-based auto-open: ArtifactCard registers itself on mount, the store
  // decides whether to auto-open. Much simpler than message-scanning.
  registerArtifactForAutoOpen: (ref: ArtifactRef) => void;
  setAutoOpenReady: () => void;
  markUserClosedForAutoOpen: () => void;
  resetAutoOpenState: () => void;

  /** Autopilot mode: 'extended_thinking' (default) or 'fast'. */
  copilotChatMode: CopilotMode;
  setCopilotChatMode: (mode: CopilotMode) => void;
  copilotModePinned: boolean;
  applyServerModeChange: (mode: CopilotMode) => void;
  clearCopilotModePin: () => void;

  /** Model tier: 'standard' (default) or 'advanced' (highest-capability). */
  copilotLlmModel: CopilotLlmModel;
  setCopilotLlmModel: (model: CopilotLlmModel) => void;

  /** Authentication route locked into the next session when it is created. */
  copilotLlmAuth: CopilotLlmAuthSelection;
  setCopilotLlmAuth: (selection: CopilotLlmAuthSelection) => void;

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

  sentMessageCount: 0,
  notifyMessageSent: () =>
    set((state) => ({ sentMessageCount: state.sentMessageCount + 1 })),

  adoptedExpertThreads: new Set<string>(),
  markExpertThreadAdopted: (expertId) =>
    set((state) => {
      const next = new Set(state.adoptedExpertThreads);
      next.add(expertId);
      return { adoptedExpertThreads: next };
    }),

  contextPanelWidth: getPersistedContextWidth(),
  artifactPanelWidth: getPersistedArtifactWidth(),
  setContextPanelWidth: (width) => {
    scheduleWidthPersist(Key.COPILOT_CONTEXT_PANEL_WIDTH, width);
    set({ contextPanelWidth: width });
  },
  setArtifactPanelWidth: (width) => {
    scheduleWidthPersist(Key.COPILOT_ARTIFACT_PANEL_WIDTH, width);
    set({ artifactPanelWidth: width });
  },

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
    activeArtifact: null,
    history: [],
    activeTab: getPersistedTab(),
    lastArtifact: null,
  },
  openArtifact: (ref, opts) =>
    set((state) => {
      const { activeArtifact, history: prevHistory } = state.artifactPanel;
      const topOfHistory = prevHistory[prevHistory.length - 1];
      const isReturningToTop = topOfHistory?.id === ref.id;
      const shouldPushHistory =
        activeArtifact != null && activeArtifact.id !== ref.id;
      const MAX_HISTORY = 25;
      const history = isReturningToTop
        ? prevHistory.slice(0, -1)
        : shouldPushHistory
          ? [...prevHistory, activeArtifact!].slice(-MAX_HISTORY)
          : prevHistory;
      if (isClient && opts?.persist !== false)
        storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "true");
      return {
        artifactPanel: {
          ...state.artifactPanel,
          isOpen: true,
          activeArtifact: ref,
          history,
        },
      };
    }),
  closeArtifactPanel: (opts) =>
    set((state) => {
      if (isClient && opts?.persist !== false)
        storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "false");
      // NOTE: deliberately does NOT set _autoOpenUserClosed. Unlike
      // toggleContextPanel (a user action), closeArtifactPanel is also the
      // programmatic collapse path — useCollapseContextPanelOnSession calls it
      // on every session entry, right before the auto-open hooks reopen the
      // panel. Suppressing auto-open here would break that collapse→reopen
      // flow. Unwanted reopen-after-user-close is already prevented per-session
      // by the auto-open hooks' own "triggered" guards.
      return {
        // Clear the preview too: the drawer is gated solely on activeArtifact,
        // so leaving it set would float a drawer with no panel behind it (and
        // a later openArtifact would wrongly treat the closed artifact as
        // back-stack history).
        artifactPanel: {
          ...state.artifactPanel,
          isOpen: false,
          activeArtifact: null,
          history: [],
          // Remember what was previewing so the sidebar toggle can bring it
          // back; closing from the tabs view remembers nothing.
          lastArtifact: state.artifactPanel.activeArtifact,
        },
      };
    }),
  clearArtifactPreview: () =>
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        activeArtifact: null,
        history: [],
      },
    })),
  resetArtifactPanel: () =>
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        activeArtifact: null,
        history: [],
        lastArtifact: null,
      },
    })),
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
  toggleContextPanel: () =>
    set((state) => {
      const nextOpen = !state.artifactPanel.isOpen;
      if (isClient)
        storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, String(nextOpen));
      // Persist the new open state. Either direction clears any open preview:
      // opening lands on the tabs view, and closing must also drop
      // activeArtifact so the artifact drawer (gated solely on activeArtifact)
      // can't stay floating after its parent panel is hidden. Closing also
      // counts as an explicit close for files auto-open.
      if (!nextOpen) _autoOpenUserClosed = true;
      return {
        artifactPanel: {
          ...state.artifactPanel,
          isOpen: nextOpen,
          activeArtifact: null,
          history: [],
        },
      };
    }),
  clearLastArtifact: () =>
    set((state) => ({
      artifactPanel: { ...state.artifactPanel, lastArtifact: null },
    })),
  toggleContextPanelTab: (tab) =>
    set((state) => {
      const { isOpen, activeTab, activeArtifact } = state.artifactPanel;
      // An open preview covers the panel, so a click there means "show me the
      // tab again" rather than "close" — only a visible matching tab closes.
      const nextOpen = !(isOpen && activeTab === tab && activeArtifact == null);
      if (isClient) {
        storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, String(nextOpen));
        storage.set(Key.COPILOT_CONTEXT_PANEL_TAB, tab);
      }
      if (!nextOpen) _autoOpenUserClosed = true;
      return {
        artifactPanel: {
          ...state.artifactPanel,
          isOpen: nextOpen,
          activeTab: tab,
          activeArtifact: null,
          history: [],
          // Closing from the tab view forgets the remembered preview, so the
          // next sidebar click reopens the tab rather than an older artifact.
          lastArtifact: nextOpen ? state.artifactPanel.lastArtifact : null,
        },
      };
    }),
  openContextPanelForFiles: () => {
    if (_autoOpenUserClosed) return;
    if (isClient) {
      storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "true");
      storage.set(Key.COPILOT_CONTEXT_PANEL_TAB, "files");
    }
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isOpen: true,
        activeTab: "files",
        activeArtifact: null,
        history: [],
      },
    }));
  },

  // Auto-open path for sessions that already have generated files: surfaces the
  // last generated file directly in the Artifact panel. Respects the user's
  // explicit close, mirroring openContextPanelForFiles' guard.
  autoOpenArtifact: (ref) => {
    if (_autoOpenUserClosed) return;
    if (isClient) storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "true");
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isOpen: true,
        activeArtifact: ref,
        history: [],
      },
    }));
  },
  // Explicit user action (the artifact panel's files button): drops the open
  // preview and hands the region to the floating files card.
  showFilesTab: () => {
    if (isClient) {
      storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "true");
      storage.set(Key.COPILOT_CONTEXT_PANEL_TAB, "files");
    }
    set((state) => ({
      artifactPanel: {
        ...state.artifactPanel,
        isOpen: true,
        activeTab: "files",
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
  copilotModePinned: false,
  applyServerModeChange: (mode) => {
    set({ copilotChatMode: mode, copilotModePinned: true });
  },
  clearCopilotModePin: () => {
    set({ copilotModePinned: false });
  },

  copilotLlmModel: (() => {
    const saved = isClient ? storage.get(Key.COPILOT_MODEL) : null;
    return saved === "advanced" ? "advanced" : "standard";
  })(),
  setCopilotLlmModel: (model) => {
    storage.set(Key.COPILOT_MODEL, model);
    set({ copilotLlmModel: model });
  },

  copilotLlmAuth: { authProvider: "platform", credentialId: null },
  setCopilotLlmAuth: (selection) => {
    set({ copilotLlmAuth: selection });
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
    storage.clean(Key.COPILOT_CONTEXT_PANEL_OPEN);
    storage.clean(Key.COPILOT_CONTEXT_PANEL_TAB);
    storage.clean(Key.COPILOT_CONTEXT_PANEL_WIDTH);
    storage.clean(Key.COPILOT_ARTIFACT_PANEL_WIDTH);
    storage.clean(Key.COPILOT_COMPLETED_SESSIONS);
    storage.clean(Key.COPILOT_DRY_RUN);
    storage.clean(Key.COPILOT_MODE);
    storage.clean(Key.COPILOT_MODEL);
    set({
      completedSessionIDs: new Set<string>(),
      contextPanelWidth: DEFAULT_PANEL_WIDTH,
      artifactPanelWidth: DEFAULT_ARTIFACT_PANEL_WIDTH,
      isSearchOpen: false,
      isNotificationsEnabled: false,
      isSoundEnabled: true,
      artifactPanel: {
        isOpen: false,
        activeArtifact: null,
        history: [],
        activeTab: "files",
        lastArtifact: null,
      },
      copilotChatMode: "extended_thinking",
      copilotLlmModel: "standard",
      copilotLlmAuth: { authProvider: "platform", credentialId: null },
      isDryRun: false,
    });
    if (isClient) {
      document.title = ORIGINAL_TITLE;
    }
  },
}));
