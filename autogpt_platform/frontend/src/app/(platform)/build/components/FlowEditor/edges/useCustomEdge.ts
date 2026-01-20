import {
  Connection as RFConnection,
  EdgeChange,
  applyEdgeChanges,
} from "@xyflow/react";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useCallback } from "react";
import { useNodeStore } from "../../../stores/nodeStore";
import { useHistoryStore } from "../../../stores/historyStore";
import { CustomEdge } from "./CustomEdge";

export const useCustomEdge = () => {
  const edges = useEdgeStore((s) => s.edges);
  const addEdge = useEdgeStore((s) => s.addEdge);
  const setEdges = useEdgeStore((s) => s.setEdges);

  const onConnect = useCallback(
    (conn: RFConnection) => {
      if (
        !conn.source ||
        !conn.target ||
        !conn.sourceHandle ||
        !conn.targetHandle
      )
        return;

      const exists = edges.some(
        (e) =>
          e.source === conn.source &&
          e.target === conn.target &&
          e.sourceHandle === conn.sourceHandle &&
          e.targetHandle === conn.targetHandle,
      );
      if (exists) return;

      const nodes = useNodeStore.getState().nodes;
      const isStatic = nodes.find((n) => n.id === conn.source)?.data
        ?.staticOutput;

      addEdge({
        source: conn.source,
        target: conn.target,
        sourceHandle: conn.sourceHandle,
        targetHandle: conn.targetHandle,
        data: {
          isStatic,
        },
      });
    },
    [edges, addEdge],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange<CustomEdge>[]) => {
      const hasRemoval = changes.some((change) => change.type === "remove");

      const prevState = hasRemoval
        ? {
            nodes: useNodeStore.getState().nodes,
            edges: edges,
          }
        : null;

      setEdges(applyEdgeChanges(changes, edges));

      if (prevState) {
        useHistoryStore.getState().pushState(prevState);
      }
    },
    [edges, setEdges],
  );

  return { edges, onConnect, onEdgesChange };
};
