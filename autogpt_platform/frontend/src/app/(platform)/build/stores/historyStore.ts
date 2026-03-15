import { create } from "zustand";
import isEqual from "lodash/isEqual";

import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { useEdgeStore } from "./edgeStore";
import { useNodeStore } from "./nodeStore";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";

type HistoryState = {
  nodes: CustomNode[];
  edges: CustomEdge[];
};

type HistoryStore = {
  past: HistoryState[];
  future: HistoryState[];
  undo: () => void;
  redo: () => void;
  initializeHistory: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
  pushState: (state: HistoryState) => void;
  clear: () => void;
};

const MAX_HISTORY = 50;

// Microtask batching state — kept outside the store to avoid triggering
// re-renders. When multiple pushState calls happen in the same synchronous
// execution (e.g. node deletion cascading to edge cleanup), only the first
// (pre-change) state is kept and committed as a single history entry.
let pendingState: HistoryState | null = null;
let batchScheduled = false;

export const useHistoryStore = create<HistoryStore>((set, get) => ({
  past: [{ nodes: [], edges: [] }],
  future: [],

  pushState: (state: HistoryState) => {
    // Keep only the first state within a microtask batch — it represents
    // the true pre-change snapshot before any cascading mutations.
    if (!pendingState) {
      pendingState = state;
    }

    if (!batchScheduled) {
      batchScheduled = true;
      queueMicrotask(() => {
        const stateToCommit = pendingState;
        pendingState = null;
        batchScheduled = false;

        if (!stateToCommit) return;

        const { past } = get();
        const lastState = past[past.length - 1];

        if (lastState && isEqual(lastState, stateToCommit)) {
          return;
        }

        const actualCurrentState = {
          nodes: useNodeStore.getState().nodes,
          edges: useEdgeStore.getState().edges,
        };

        if (isEqual(stateToCommit, actualCurrentState)) {
          return;
        }

        set((prev) => ({
          past: [...prev.past.slice(-MAX_HISTORY + 1), stateToCommit],
          future: [],
        }));
      });
    }
  },

  initializeHistory: () => {
    pendingState = null;

    const currentNodes = useNodeStore.getState().nodes;
    const currentEdges = useEdgeStore.getState().edges;

    set({
      past: [{ nodes: currentNodes, edges: currentEdges }],
      future: [],
    });
  },

  undo: () => {
    const { past, future } = get();
    if (past.length === 0) return;

    const actualCurrentState = {
      nodes: useNodeStore.getState().nodes,
      edges: useEdgeStore.getState().edges,
    };

    const previousState = past[past.length - 1];

    if (isEqual(actualCurrentState, previousState)) {
      return;
    }

    useNodeStore.getState().setNodes(previousState.nodes);
    useEdgeStore.getState().setEdges(previousState.edges);

    set({
      past: past.length > 1 ? past.slice(0, -1) : past,
      future: [actualCurrentState, ...future],
    });
  },

  redo: () => {
    const { past, future } = get();
    if (future.length === 0) return;

    const actualCurrentState = {
      nodes: useNodeStore.getState().nodes,
      edges: useEdgeStore.getState().edges,
    };

    const nextState = future[0];

    useNodeStore.getState().setNodes(nextState.nodes);
    useEdgeStore.getState().setEdges(nextState.edges);

    const lastPast = past[past.length - 1];
    const shouldPushToPast =
      !lastPast || !isEqual(actualCurrentState, lastPast);

    set({
      past: shouldPushToPast ? [...past, actualCurrentState] : past,
      future: future.slice(1),
    });
  },

  canUndo: () => {
    const { past } = get();
    if (past.length === 0) return false;

    const actualCurrentState = {
      nodes: useNodeStore.getState().nodes,
      edges: useEdgeStore.getState().edges,
    };
    return !isEqual(actualCurrentState, past[past.length - 1]);
  },
  canRedo: () => get().future.length > 0,

  clear: () => {
    pendingState = null;
    set({ past: [{ nodes: [], edges: [] }], future: [] });
  },
}));
