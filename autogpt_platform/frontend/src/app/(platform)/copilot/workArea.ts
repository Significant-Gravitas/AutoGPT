import type { UIMessage } from "ai";
import { create } from "zustand";
import type { RecipientOption } from "./components/ChatInput/components/RecipientChip";
import type { ArtifactRef } from "./store";

/** One tab of the chat's right sidebar. */
export type WorkTab =
  | { key: string; kind: "task"; taskID: string; title: string }
  | { key: string; kind: "artifact"; artifact: ArtifactRef }
  | { key: string; kind: "start" }
  | { key: string; kind: "profile" }
  | { key: string; kind: "brief"; title: string }
  | {
      key: string;
      kind: "workflow";
      graphID: string;
      graphVersion: number | null;
      /** The run on show; none shows the newest. */
      executionID: string | null;
      title: string;
    };

/** What opens a workflow's tab: the workflow, and the run to show. A null
 *  run shows the list of runs; leaving it out keeps what the tab shows. */
export interface WorkflowRunRef {
  graphID: string;
  graphVersion?: number | null;
  executionID?: string | null;
  title: string;
}

/** The Files, Artifacts or Integrations face the store opens. */
export const CONTEXT_TAB_KEY = "context";
/** The chat's computer, while it is open. */
export const COMPUTER_TAB_KEY = "computer";
/** A new tab: the start page that lists what can be opened. */
export const START_TAB_KEY = "start";
/** Who Home's composer is talking to: Otto or an expert. */
export const PROFILE_TAB_KEY = "profile";
/** The day's brief, opened from Home. */
export const BRIEF_TAB_KEY = "brief";

export function taskTabKey(taskID: string) {
  return `task:${taskID}`;
}

export function workflowTabKey(graphID: string) {
  return `workflow:${graphID}`;
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
  /** Open a workflow's runs as a tab, on the given run when there is one;
   *  an open tab of the same workflow moves to that run. */
  openWorkflowRun: (ref: WorkflowRunRef) => void;
  /** A task's tab takes the task's own name, once it is known or changes. */
  nameTask: (taskID: string, title: string) => void;
  /** A workflow's tab takes the workflow's own name, once it is known. */
  nameWorkflow: (graphID: string, title: string) => void;
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
  /** The sidebar's own close button: fold it away, tabs kept. */
  fold: () => void;
  /** Whose profile the Profile tab shows; Home keeps it in step with its
   *  composer. */
  profileRecipient: RecipientOption | null;
  setProfileRecipient: (recipient: RecipientOption | null) => void;
  /** The profile of who Home is talking to, in front. */
  openProfile: () => void;
  /** Home's sidebar always offers the Profile tab beside whatever else is
   *  open; with nothing open it stays closed. */
  offerProfile: () => void;
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
  openWorkflowRun: (ref) =>
    set((state) => {
      const key = workflowTabKey(ref.graphID);
      const open = state.tabs.find((tab) => tab.key === key);
      const next: WorkTab = {
        key,
        kind: "workflow",
        graphID: ref.graphID,
        graphVersion:
          ref.graphVersion ??
          (open?.kind === "workflow" ? open.graphVersion : null),
        executionID:
          ref.executionID !== undefined
            ? ref.executionID
            : open?.kind === "workflow"
              ? open.executionID
              : null,
        title: ref.title || (open?.kind === "workflow" ? open.title : ""),
      };
      return {
        tabs: open
          ? state.tabs.map((tab) => (tab.key === key ? next : tab))
          : [...state.tabs, next],
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
  nameWorkflow: (graphID, title) =>
    set((state) => {
      const key = workflowTabKey(graphID);
      const tab = state.tabs.find((entry) => entry.key === key);
      if (!tab || tab.kind !== "workflow" || tab.title === title) return state;
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
  fold: () => set({ isFolded: true }),
  profileRecipient: null,
  setProfileRecipient: (recipient) =>
    set((state) => {
      const current = state.profileRecipient;
      const isSame =
        current?.id === recipient?.id && current?.name === recipient?.name;
      return isSame && current !== null
        ? state
        : { profileRecipient: recipient };
    }),
  offerProfile: () =>
    set((state) => {
      const isOffered = state.tabs.some((tab) => tab.kind === "profile");
      if (isOffered || state.tabs.length === 0) return state;
      return {
        tabs: [...state.tabs, { key: PROFILE_TAB_KEY, kind: "profile" }],
      };
    }),
  openProfile: () =>
    set((state) => ({
      // eslint-disable-next-line no-restricted-syntax -- a tab's own key, not a keyboard event's
      tabs: state.tabs.some((tab) => tab.key === PROFILE_TAB_KEY)
        ? state.tabs
        : [...state.tabs, { key: PROFILE_TAB_KEY, kind: "profile" }],
      activeKey: PROFILE_TAB_KEY,
      isFolded: false,
    })),
  reportTabCount: (count) =>
    set((state) => (state.tabCount === count ? state : { tabCount: count })),
  reset: () => set({ tabs: [], activeKey: null, isFolded: false, tabCount: 0 }),
  chatSessionID: null,
  chatMessages: [],
  publishChat: (sessionID, messages) =>
    set({ chatSessionID: sessionID, chatMessages: messages }),
}));
