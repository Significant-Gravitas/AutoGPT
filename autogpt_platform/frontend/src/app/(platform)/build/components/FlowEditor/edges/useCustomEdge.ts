import { Connection as RFConnection, EdgeChange } from "@xyflow/react";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useCallback } from "react";
import { useNodeStore } from "../../../stores/nodeStore";

export const useCustomEdge = () => {
  const edges = useEdgeStore((s) => s.edges);
  const addEdge = useEdgeStore((s) => s.addEdge);
  const removeEdge = useEdgeStore((s) => s.removeEdge);

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
    (changes: EdgeChange[]) => {
      changes.forEach((change) => {
        if (change.type === "remove") {
          removeEdge(change.id);
        }
      });
    },
    [removeEdge],
  );

  return { edges, onConnect, onEdgesChange };
};
