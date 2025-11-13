import { create } from "zustand";
import { Link } from "@/app/api/__generated__/models/link";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";
import { customEdgeToLink, linkToCustomEdge } from "../components/helper";
import { MarkerType } from "@xyflow/react";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";

type EdgeStore = {
  edges: CustomEdge[];

  setEdges: (edges: CustomEdge[]) => void;
  addEdge: (edge: Omit<CustomEdge, "id"> & { id?: string }) => CustomEdge;
  removeEdge: (edgeId: string) => void;
  upsertMany: (edges: CustomEdge[]) => void;

  getNodeEdges: (nodeId: string) => CustomEdge[];
  isInputConnected: (nodeId: string, handle: string) => boolean;
  isOutputConnected: (nodeId: string, handle: string) => boolean;
  getBackendLinks: () => Link[];
  addLinks: (links: Link[]) => void;

  getAllHandleIdsOfANode: (nodeId: string) => string[];

  updateEdgeBeads: (
    targetNodeId: string,
    executionResult: NodeExecutionResult,
  ) => void;
  resetEdgeBeads: () => void;
};

function makeEdgeId(edge: Omit<CustomEdge, "id">) {
  return `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`;
}

export const useEdgeStore = create<EdgeStore>((set, get) => ({
  edges: [],

  setEdges: (edges) => set({ edges }),

  addEdge: (edge) => {
    const id = edge.id || makeEdgeId(edge);
    const newEdge: CustomEdge = {
      type: "custom" as const,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        strokeWidth: 2,
        color: "#555",
      },
      ...edge,
      id,
    };

    set((state) => {
      const exists = state.edges.some(
        (e) =>
          e.source === newEdge.source &&
          e.target === newEdge.target &&
          e.sourceHandle === newEdge.sourceHandle &&
          e.targetHandle === newEdge.targetHandle,
      );
      if (exists) return state;
      return { edges: [...state.edges, newEdge] };
    });

    return newEdge;
  },

  removeEdge: (edgeId) =>
    set((state) => ({
      edges: state.edges.filter((e) => e.id !== edgeId),
    })),

  upsertMany: (edges) =>
    set((state) => {
      const byKey = new Map(state.edges.map((e) => [e.id, e]));
      edges.forEach((e) => {
        byKey.set(e.id, e);
      });
      return { edges: Array.from(byKey.values()) };
    }),

  getNodeEdges: (nodeId) =>
    get().edges.filter((e) => e.source === nodeId || e.target === nodeId),

  isInputConnected: (nodeId, handle) =>
    get().edges.some((e) => e.target === nodeId && e.targetHandle === handle),

  isOutputConnected: (nodeId, handle) =>
    get().edges.some((e) => e.source === nodeId && e.sourceHandle === handle),

  getBackendLinks: () => get().edges.map(customEdgeToLink),

  addLinks: (links) => {
    links.forEach((link) => {
      get().addEdge(linkToCustomEdge(link));
    });
  },

  getAllHandleIdsOfANode: (nodeId) =>
    get()
      .edges.filter((e) => e.target === nodeId)
      .map((e) => e.targetHandle || ""),

  updateEdgeBeads: (
    targetNodeId: string,
    executionResult: NodeExecutionResult,
  ) => {
    set((state) => ({
      edges: state.edges.map((edge) => {
        if (edge.target !== targetNodeId) {
          return edge;
        }

        const beadData =
          edge.data?.beadData ??
          new Map<string, NodeExecutionResult["status"]>();

        if (
          edge.targetHandle &&
          edge.targetHandle in executionResult.input_data
        ) {
          beadData.set(executionResult.node_exec_id, executionResult.status);
        }

        let beadUp = 0;
        let beadDown = 0;

        beadData.forEach((status) => {
          beadUp++;
          if (status !== "INCOMPLETE") {
            beadDown++;
          }
        });

        if (edge.data?.isStatic && beadUp > 0) {
          beadUp = beadDown + 1;
        }

        return {
          ...edge,
          data: {
            ...edge.data,
            beadUp,
            beadDown,
            beadData,
          },
        };
      }),
    }));
  },

  resetEdgeBeads: () => {
    set((state) => ({
      edges: state.edges.map((edge) => ({
        ...edge,
        data: {
          ...edge.data,
          beadUp: 0,
          beadDown: 0,
          beadData: new Map(),
        },
      })),
    }));
  },
}));
