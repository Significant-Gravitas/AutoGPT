import { create } from "zustand";
import isEqual from "lodash/isEqual";

import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { useEdgeStore } from "./edgeStore";
import { useNodeStore } from "./nodeStore";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";

type HistoryState = {
  nodes: CustomNode[];
  edges: CustomEdge[];
  nodeCounter?: number;
};

type HistoryStore = {
  past: HistoryState[];
  future: HistoryState[];
  isApplyingHistory: boolean;
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

function normalizeHistoryState(s: HistoryState): HistoryState {
  return { nodes: s.nodes, edges: s.edges, nodeCounter: s.nodeCounter ?? 0 };
}

export const useHistoryStore = create<HistoryStore>((set, get) => ({
  past: [{ nodes: [], edges: [], nodeCounter: 0 }],
  future: [],
  isApplyingHistory: false,

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

        if (
          lastState &&
          isEqual(
            normalizeHistoryState(lastState),
            normalizeHistoryState(stateToCommit),
          )
        ) {
          return;
        }

        const actualCurrentState: HistoryState = {
          nodes: useNodeStore.getState().nodes,
          edges: useEdgeStore.getState().edges,
          nodeCounter: useNodeStore.getState().nodeCounter,
        };

        if (
          isEqual(
            normalizeHistoryState(stateToCommit),
            normalizeHistoryState(actualCurrentState),
          )
        ) {
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
    const nodeCounter = useNodeStore.getState().nodeCounter;

    set({
      past: [{ nodes: currentNodes, edges: currentEdges, nodeCounter }],
      future: [],
      isApplyingHistory: false,
    });
  },

  undo: () => {
    const { past, future } = get();
    if (past.length === 0) return;

    const actualCurrentState: HistoryState = {
      nodes: useNodeStore.getState().nodes,
      edges: useEdgeStore.getState().edges,
      nodeCounter: useNodeStore.getState().nodeCounter,
    };

    const previousState = past[past.length - 1];

    if (
      isEqual(
        normalizeHistoryState(actualCurrentState),
        normalizeHistoryState(previousState),
      )
    ) {
      return;
    }

    set({ isApplyingHistory: true });
    useNodeStore.getState().setNodes(previousState.nodes);
    useEdgeStore.getState().setEdges(previousState.edges);
    if (previousState.nodeCounter !== undefined) {
      useNodeStore.setState({ nodeCounter: previousState.nodeCounter });
    }

    set({
      past: past.length > 1 ? past.slice(0, -1) : past,
      future: [actualCurrentState, ...future],
      isApplyingHistory: false,
    });
  },

  redo: () => {
    const { past, future } = get();
    if (future.length === 0) return;

    const actualCurrentState: HistoryState = {
      nodes: useNodeStore.getState().nodes,
      edges: useEdgeStore.getState().edges,
      nodeCounter: useNodeStore.getState().nodeCounter,
    };

    const nextState = future[0];

    set({ isApplyingHistory: true });
    useNodeStore.getState().setNodes(nextState.nodes);
    useEdgeStore.getState().setEdges(nextState.edges);
    if (nextState.nodeCounter !== undefined) {
      useNodeStore.setState({ nodeCounter: nextState.nodeCounter });
    }

    const lastPast = past[past.length - 1];
    const shouldPushToPast =
      !lastPast ||
      !isEqual(
        normalizeHistoryState(actualCurrentState),
        normalizeHistoryState(lastPast),
      );

    set({
      past: shouldPushToPast ? [...past, actualCurrentState] : past,
      future: future.slice(1),
      isApplyingHistory: false,
    });
  },

  canUndo: () => {
    const { past } = get();
    if (past.length === 0) return false;

    const actualCurrentState: HistoryState = {
      nodes: useNodeStore.getState().nodes,
      edges: useEdgeStore.getState().edges,
      nodeCounter: useNodeStore.getState().nodeCounter,
    };
    return !isEqual(
      normalizeHistoryState(actualCurrentState),
      normalizeHistoryState(past[past.length - 1]),
    );
  },
  canRedo: () => get().future.length > 0,

  clear: () => {
    pendingState = null;
    set({ past: [{ nodes: [], edges: [], nodeCounter: 0 }], future: [] });
  },
}));
