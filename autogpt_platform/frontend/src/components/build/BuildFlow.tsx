"use client";

import React, { useCallback, useMemo, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  Connection,
  Edge,
  Node,
  NodeChange,
  EdgeChange,
} from "@xyflow/react";
import { Undo2, Redo2, RefreshCw } from "lucide-react";
import ControlPanel from "@/components/edit/control/ControlPanel";
import useUndoRedo from "@/hooks/build/useUndoRedo";

export interface CanvasProps {
  id?: string;
  initialNodes?: Node<any>[];
  initialEdges?: Edge<any>[];
  readOnly?: boolean;
}

const CanvasInner = ({
  id = "canvas",
  initialNodes = [],
  initialEdges = [],
  readOnly = false,
}: CanvasProps) => {
  const { current, addState, undo, redo, canUndo, canRedo, reset } =
    useUndoRedo({
      nodes: initialNodes,
      edges: initialEdges,
    });

  const [tempNodes, setTempNodes] = useState<Node<any>[]>(current.nodes);
  const [isDragging, setIsDragging] = useState(false);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const updatedNodes = applyNodeChanges(changes, tempNodes);
      setTempNodes(updatedNodes);

      // Only save state for significant changes
      const hasSignificantChanges = changes.some(
        (change) =>
          change.type === "position" ||
          change.type === "remove" ||
          change.type === "add",
      );

      if (!isDragging && hasSignificantChanges) {
        const cleanNodes = updatedNodes.map((node) => {
          const { selected, measured, dragging, ...cleanNode } = node;
          return cleanNode;
        });
        addState({ nodes: cleanNodes, edges: current.edges });
      }
    },
    [tempNodes, current.edges, addState, isDragging],
  );

  const onNodeDragStart = useCallback(() => {
    setIsDragging(true);
  }, []);

  const onNodeDragStop = useCallback(() => {
    setIsDragging(false);
    // Clean nodes before saving
    const cleanNodes = tempNodes.map((node) => {
      const { selected, measured, dragging, ...cleanNode } = node;
      return cleanNode;
    });

    if (JSON.stringify(cleanNodes) !== JSON.stringify(current.nodes)) {
      addState({ nodes: cleanNodes, edges: current.edges });
    }
  }, [tempNodes, current.nodes, current.edges, addState]);

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const updatedEdges = applyEdgeChanges(changes, current.edges);
      if (changes.length > 0) {
        addState({ nodes: current.nodes, edges: updatedEdges });
      }
    },
    [current.nodes, current.edges, addState],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const updatedEdges = addEdge(connection, current.edges);
      addState({ nodes: current.nodes, edges: updatedEdges });
    },
    [current.nodes, current.edges, addState],
  );

  // Sync tempNodes when current.nodes changes (e.g., after undo/redo)
  React.useEffect(() => {
    setTempNodes(current.nodes);
  }, [current.nodes]);

  const controls = useMemo(
    () => [
      {
        icon: <Undo2 className="h-4 w-4" />,
        label: "Undo",
        disabled: !canUndo,
        onClick: undo,
      },
      {
        icon: <Redo2 className="h-4 w-4" />,
        label: "Redo",
        disabled: !canRedo,
        onClick: redo,
      },
      {
        icon: <RefreshCw className="h-4 w-4" />,
        label: "Reset",
        onClick: reset,
      },
    ],
    [canUndo, canRedo, undo, redo, reset],
  );

  return (
    <div className="relative flex h-full w-full">
      <ControlPanel controls={controls} className="z-10" />
      <div className="flex-1">
        <ReactFlow
          id={id}
          nodes={tempNodes}
          edges={current.edges}
          onNodesChange={onNodesChange}
          onNodeDragStart={onNodeDragStart}
          onNodeDragStop={onNodeDragStop}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodesDraggable={!readOnly}
          nodesConnectable={!readOnly}
          fitView
          className={readOnly ? "read-only-mode" : ""}
        >
          <Controls />
          <Background gap={16} color="#ddd" />
        </ReactFlow>
      </div>
    </div>
  );
};

const BuildFlow = (props: CanvasProps) => (
  <ReactFlowProvider>
    <CanvasInner {...props} />
  </ReactFlowProvider>
);

export default BuildFlow;
