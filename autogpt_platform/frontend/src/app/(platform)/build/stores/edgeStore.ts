import { create } from "zustand";
import { convertConnectionsToBackendLinks } from "../components/FlowEditor/edges/helpers";

export type Connection = {
  edge_id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
};

type EdgeStore = {
  connections: Connection[];

  setConnections: (connections: Connection[]) => void;
  addConnection: (
    conn: Omit<Connection, "edge_id"> & { edge_id?: string },
  ) => Connection;
  removeConnection: (edge_id: string) => void;
  upsertMany: (conns: Connection[]) => void;

  getNodeConnections: (nodeId: string) => Connection[];
  isInputConnected: (nodeId: string, handle: string) => boolean;
  isOutputConnected: (nodeId: string, handle: string) => boolean;
};

function makeEdgeId(conn: Omit<Connection, "edge_id">) {
  return `${conn.source}:${conn.sourceHandle}->${conn.target}:${conn.targetHandle}`;
}

export const useEdgeStore = create<EdgeStore>((set, get) => ({
  connections: [],

  setConnections: (connections) => set({ connections }),

  addConnection: (conn) => {
    const edge_id = conn.edge_id || makeEdgeId(conn);
    const newConn: Connection = { edge_id, ...conn };

    set((state) => {
      const exists = state.connections.some(
        (c) =>
          c.source === newConn.source &&
          c.target === newConn.target &&
          c.sourceHandle === newConn.sourceHandle &&
          c.targetHandle === newConn.targetHandle,
      );
      if (exists) return state;
      return { connections: [...state.connections, newConn] };
    });

    return { edge_id, ...conn };
  },

  removeConnection: (edge_id) =>
    set((state) => ({
      connections: state.connections.filter((c) => c.edge_id !== edge_id),
    })),

  upsertMany: (conns) =>
    set((state) => {
      const byKey = new Map(state.connections.map((c) => [c.edge_id, c]));
      conns.forEach((c) => {
        byKey.set(c.edge_id, c);
      });
      return { connections: Array.from(byKey.values()) };
    }),

  getNodeConnections: (nodeId) =>
    get().connections.filter((c) => c.source === nodeId || c.target === nodeId),

  isInputConnected: (nodeId, handle) =>
    get().connections.some(
      (c) => c.target === nodeId && c.targetHandle === handle,
    ),

  isOutputConnected: (nodeId, handle) =>
    get().connections.some(
      (c) => c.source === nodeId && c.sourceHandle === handle,
    ),
  getBackendLinks: () => convertConnectionsToBackendLinks(get().connections),
}));
