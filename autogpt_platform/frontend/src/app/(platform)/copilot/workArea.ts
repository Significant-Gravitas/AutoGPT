import type { UIMessage } from "ai";
import { create } from "zustand";
import type { ArtifactRef } from "./store";

/** One tab of the chat's right sidebar. */
export type WorkTab =
  | { key: string; kind: "task"; taskID: string; title: string }
  | { key: string; kind: "artifact"; artifact: ArtifactRef }
  | { key: string; kind: "start" }
  | { key: string; kind: "brief"; title: string };

/** The Files, Artifacts or Integrations face the store opens. */
export const CONTEXT_TAB_KEY = "context";
/** The chat's computer, while it is open. */
export const COMPUTER_TAB_KEY = "computer";
/** A new tab: the start page that lists what can be opened. */
export const START_TAB_KEY = "start";
/** The day's brief, opened from Home. */
export const BRIEF_TAB_KEY = "brief";

export function taskTabKey(taskID: string) {
  return `task:${taskID}`;
}

export function artifactTabKey(artifactID: string) {
  return `artifact:${artifactID}`;
}

interface WorkAreaState {
  tabs: WorkTab[];
  activeKey: string | null;
  /** Folded away with its tabs kept, until the open button brings it back. */
  isFolded: boolean;
  /** How many tabs the sidebar has, the chat's own faces included. */
  tabCount: number;
  /** Open a task as a tab, or come back to it when it is already open. */
  openTask: (taskID: string, title: string) => void;
  /** A task's tab takes the task's own name, once it is known or changes. */
  nameTask: (taskID: string, title: string) => void;
  /** The store showed an artifact: give it a tab and bring it forward. A new
   *  version of an open file takes its tab over. */
  showArtifact: (artifact: ArtifactRef) => void;
  /** A new tab, on the start page. */
  openStart: () => void;
  /** The day's brief, in front; it keeps the first place among the tabs. */
  openBrief: (title: string) => void;
  activate: (key: string) => void;
  /** Close a tab; the one before it comes forward. */
  close: (key: string) => void;
  /** The open button: fold the sidebar away, or bring it back; with nothing
   *  open it comes back on a new tab. */
  toggle: () => void;
  unfold: () => void;
  reportTabCount: (count: number) => void;
  reset: () => void;
  /** The open chat's messages, as the chat itself renders them. Both chat
   *  engines publish here, so the sidebar reads one place. */
  chatSessionID: string | null;
  chatMessages: UIMessage[];
  publishChat: (sessionID: string, messages: UIMessage[]) => void;
}

export const useWorkAreaStore = create<WorkAreaState>((set) => ({
  tabs: [],
  activeKey: null,
  isFolded: false,
  tabCount: 0,
  openTask: (taskID, title) =>
    set((state) => {
      const key = taskTabKey(taskID);
      const exists = state.tabs.some((tab) => tab.key === key);
      return {
        tabs: exists
          ? state.tabs
          : [...state.tabs, { key, kind: "task", taskID, title }],
        activeKey: key,
        isFolded: false,
      };
    }),
  nameTask: (taskID, title) =>
    set((state) => {
      const key = taskTabKey(taskID);
      const tab = state.tabs.find((entry) => entry.key === key);
      if (!tab || tab.kind !== "task" || tab.title === title) return state;
      return {
        tabs: state.tabs.map((entry) =>
          entry === tab ? { ...tab, title } : entry,
        ),
      };
    }),
  showArtifact: (artifact) =>
    set((state) => {
      const key = artifactTabKey(artifact.documentId ?? artifact.id);
      const next: WorkTab = { key, kind: "artifact", artifact };
      const exists = state.tabs.some((tab) => tab.key === key);
      return {
        tabs: exists
          ? state.tabs.map((tab) => (tab.key === key ? next : tab))
          : [...state.tabs, next],
        activeKey: key,
        isFolded: false,
      };
    }),
  openStart: () =>
    set((state) => ({
      // eslint-disable-next-line no-restricted-syntax -- a tab's own key, not a keyboard event's
      tabs: state.tabs.some((tab) => tab.key === START_TAB_KEY)
        ? state.tabs
        : [...state.tabs, { key: START_TAB_KEY, kind: "start" }],
      activeKey: START_TAB_KEY,
      isFolded: false,
    })),
  openBrief: (title) =>
    set((state) => {
      const key = BRIEF_TAB_KEY;
      const exists = state.tabs.some((tab) => tab.key === key);
      return {
        tabs: exists
          ? state.tabs
          : [{ key, kind: "brief", title }, ...state.tabs],
        activeKey: key,
        isFolded: false,
      };
    }),
  activate: (key) => set({ activeKey: key }),
  close: (key) =>
    set((state) => {
      const index = state.tabs.findIndex((tab) => tab.key === key);
      if (index === -1) return state;
      const tabs = state.tabs.filter((tab) => tab.key !== key);
      const neighbour = tabs[Math.max(0, index - 1)] ?? null;
      return {
        tabs,
        activeKey:
          state.activeKey === key ? (neighbour?.key ?? null) : state.activeKey,
      };
    }),
  toggle: () =>
    set((state) => {
      if (state.tabCount > 0) return { isFolded: !state.isFolded };
      return {
        tabs: [{ key: START_TAB_KEY, kind: "start" }],
        activeKey: START_TAB_KEY,
        isFolded: false,
      };
    }),
  unfold: () => set({ isFolded: false }),
  reportTabCount: (count) =>
    set((state) => (state.tabCount === count ? state : { tabCount: count })),
  reset: () => set({ tabs: [], activeKey: null, isFolded: false, tabCount: 0 }),
  chatSessionID: null,
  chatMessages: [],
  publishChat: (sessionID, messages) =>
    set({ chatSessionID: sessionID, chatMessages: messages }),
}));
