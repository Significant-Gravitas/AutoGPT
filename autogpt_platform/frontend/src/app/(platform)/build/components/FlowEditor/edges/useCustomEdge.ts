import {
  Connection as RFConnection,
  Edge as RFEdge,
  MarkerType,
  EdgeChange,
} from "@xyflow/react";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useCallback, useMemo } from "react";

export const useCustomEdge = () => {
  const connections = useEdgeStore((s) => s.connections);
  const addConnection = useEdgeStore((s) => s.addConnection);
  const removeConnection = useEdgeStore((s) => s.removeConnection);

  const edges: RFEdge[] = useMemo(
    () =>
      connections.map((c) => ({
        id: c.edge_id,
        type: "custom",
        source: c.source,
        target: c.target,
        sourceHandle: c.sourceHandle,
        targetHandle: c.targetHandle,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          strokeWidth: 2,
          color: "#555",
        },
      })),
    [connections],
  );

  const onConnect = useCallback(
    (conn: RFConnection) => {
      if (
        !conn.source ||
        !conn.target ||
        !conn.sourceHandle ||
        !conn.targetHandle
      )
        return;
      const exists = connections.some(
        (c) =>
          c.source === conn.source &&
          c.target === conn.target &&
          c.sourceHandle === conn.sourceHandle &&
          c.targetHandle === conn.targetHandle,
      );
      if (exists) return;
      addConnection({
        source: conn.source,
        target: conn.target,
        sourceHandle: conn.sourceHandle,
        targetHandle: conn.targetHandle,
      });
    },
    [connections, addConnection],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      changes.forEach((ch) => {
        if (ch.type === "remove") removeConnection(ch.id);
      });
    },
    [removeConnection],
  );

  return { edges, onConnect, onEdgesChange };
};
