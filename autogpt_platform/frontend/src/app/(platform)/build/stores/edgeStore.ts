import { create } from "zustand";
import { Link } from "@/app/api/__generated__/models/link";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";
import { customEdgeToLink, linkToCustomEdge } from "../components/helper";
import { MarkerType } from "@xyflow/react";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { cleanUpHandleId } from "@/components/renderers/InputRenderer/helpers";
import { useHistoryStore } from "./historyStore";
import { useNodeStore } from "./nodeStore";

type EdgeStore = {
  edges: CustomEdge[];

  setEdges: (edges: CustomEdge[]) => void;
  addEdge: (edge: Omit<CustomEdge, "id"> & { id?: string }) => CustomEdge;
  removeEdge: (edgeId: string) => void;
  upsertMany: (edges: CustomEdge[]) => void;

  removeEdgesByHandlePrefix: (nodeId: string, handlePrefix: string) => void;

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

    const exists = get().edges.some(
      (e) =>
        e.source === newEdge.source &&
        e.target === newEdge.target &&
        e.sourceHandle === newEdge.sourceHandle &&
        e.targetHandle === newEdge.targetHandle,
    );
    if (exists) return newEdge;
    const prevState = {
      nodes: useNodeStore.getState().nodes,
      edges: get().edges,
    };

    set((state) => ({ edges: [...state.edges, newEdge] }));
    useHistoryStore.getState().pushState(prevState);

    return newEdge;
  },

  removeEdge: (edgeId) => {
    const prevState = {
      nodes: useNodeStore.getState().nodes,
      edges: get().edges,
    };

    set((state) => ({
      edges: state.edges.filter((e) => e.id !== edgeId),
    }));
    useHistoryStore.getState().pushState(prevState);
  },

  upsertMany: (edges) =>
    set((state) => {
      const byKey = new Map(state.edges.map((e) => [e.id, e]));
      edges.forEach((e) => {
        byKey.set(e.id, e);
      });
      return { edges: Array.from(byKey.values()) };
    }),

  removeEdgesByHandlePrefix: (nodeId, handlePrefix) =>
    set((state) => ({
      edges: state.edges.filter(
        (e) =>
          !(
            e.target === nodeId &&
            e.targetHandle &&
            e.targetHandle.startsWith(handlePrefix)
          ),
      ),
    })),

  getNodeEdges: (nodeId) =>
    get().edges.filter((e) => e.source === nodeId || e.target === nodeId),

  isInputConnected: (nodeId, handle) => {
    const cleanedHandle = cleanUpHandleId(handle);
    return get().edges.some(
      (e) => e.target === nodeId && e.targetHandle === cleanedHandle,
    );
  },

  isOutputConnected: (nodeId, handle) =>
    get().edges.some((e) => e.source === nodeId && e.sourceHandle === handle),

  getBackendLinks: () => {
    // Filter out edges referencing non-existent nodes before converting to links
    const nodeIds = new Set(useNodeStore.getState().nodes.map((n) => n.id));
    const validEdges = get().edges.filter((edge) => {
      const isValid = nodeIds.has(edge.source) && nodeIds.has(edge.target);
      if (!isValid) {
        console.warn(
          `[EdgeStore] Filtering out invalid edge during save: source=${edge.source}, target=${edge.target}`,
        );
      }
      return isValid;
    });
    return validEdges.map(customEdgeToLink);
  },

  addLinks: (links) => {
    // Get current node IDs to validate links
    const nodeIds = new Set(useNodeStore.getState().nodes.map((n) => n.id));

    links.forEach((link) => {
      // Skip invalid links (orphan edges referencing non-existent nodes)
      if (!nodeIds.has(link.source_id) || !nodeIds.has(link.sink_id)) {
        console.warn(
          `[EdgeStore] Skipping invalid link: source=${link.source_id}, sink=${link.sink_id} - node(s) not found`,
        );
        return;
      }
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
    set((state) => {
      let hasChanges = false;

      const newEdges = state.edges.map((edge) => {
        if (edge.target !== targetNodeId) {
          return edge;
        }

        const beadData = new Map(edge.data?.beadData ?? new Map());

        const inputValue = edge.targetHandle
          ? executionResult.input_data[edge.targetHandle]
          : undefined;

        if (inputValue !== undefined && inputValue !== null) {
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

        if (edge.data?.beadUp === beadUp && edge.data?.beadDown === beadDown) {
          return edge;
        }

        hasChanges = true;
        return {
          ...edge,
          data: {
            ...edge.data,
            beadUp,
            beadDown,
            beadData,
          },
        };
      });

      return hasChanges ? { edges: newEdges } : state;
    });
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
