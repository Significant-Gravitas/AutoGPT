import { create } from "zustand";
import isEqual from "lodash/isEqual";

import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { Connection, useEdgeStore } from "./edgeStore";
import { useNodeStore } from "./nodeStore";

type HistoryState = {
  nodes: CustomNode[];
  connections: Connection[];
};

type HistoryStore = {
  past: HistoryState[];
  future: HistoryState[];
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
  pushState: (state: HistoryState) => void;
  clear: () => void;
};

const MAX_HISTORY = 50;

export const useHistoryStore = create<HistoryStore>((set, get) => ({
  past: [{ nodes: [], connections: [] }],
  future: [],

  pushState: (state: HistoryState) => {
    const { past } = get();
    const lastState = past[past.length - 1];

    if (lastState && isEqual(lastState, state)) {
      return;
    }

    set((prev) => ({
      past: [...prev.past.slice(-MAX_HISTORY + 1), state],
      future: [],
    }));
  },

  undo: () => {
    const { past, future } = get();
    if (past.length <= 1) return;

    const currentState = past[past.length - 1];

    const previousState = past[past.length - 2];

    useNodeStore.getState().setNodes(previousState.nodes);
    useEdgeStore.getState().setConnections(previousState.connections);

    set({
      past: past.slice(0, -1),
      future: [currentState, ...future],
    });
  },

  redo: () => {
    const { past, future } = get();
    if (future.length === 0) return;

    const nextState = future[0];

    useNodeStore.getState().setNodes(nextState.nodes);
    useEdgeStore.getState().setConnections(nextState.connections);

    set({
      past: [...past, nextState],
      future: future.slice(1),
    });
  },

  canUndo: () => get().past.length > 1,
  canRedo: () => get().future.length > 0,

  clear: () => set({ past: [{ nodes: [], connections: [] }], future: [] }),
}));
