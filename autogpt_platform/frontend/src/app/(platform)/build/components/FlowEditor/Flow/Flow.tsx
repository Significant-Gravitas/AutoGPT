import { ReactFlow, Background } from "@xyflow/react";
import NewControlPanel from "../../NewControlPanel/NewControlPanel";
import CustomEdge from "../edges/CustomEdge";
import { useFlow } from "./useFlow";
import { useShallow } from "zustand/react/shallow";
import { useNodeStore } from "../../../stores/nodeStore";
import { useMemo, useEffect } from "react";
import { CustomNode } from "../nodes/CustomNode/CustomNode";
import { useCustomEdge } from "../edges/useCustomEdge";
import { useFlowRealtime } from "./useFlowRealtime";
import { GraphLoadingBox } from "./components/GraphLoadingBox";
import { BuilderActions } from "../../BuilderActions/BuilderActions";
import { RunningBackground } from "./components/RunningBackground";
import { useGraphStore } from "../../../stores/graphStore";
import { useCopyPaste } from "./useCopyPaste";
import { CustomControls } from "./components/CustomControl";

export const Flow = () => {
  const nodes = useNodeStore(useShallow((state) => state.nodes));
  const onNodesChange = useNodeStore(
    useShallow((state) => state.onNodesChange),
  );
  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);
  const edgeTypes = useMemo(() => ({ custom: CustomEdge }), []);
  const { edges, onConnect, onEdgesChange } = useCustomEdge();

  // We use this hook to load the graph and convert them into custom nodes and edges.
  const { onDragOver, onDrop, isFlowContentLoading, isLocked, setIsLocked } =
    useFlow();

  // This hook is used for websocket realtime updates.
  useFlowRealtime();

  // Copy/paste functionality
  const handleCopyPaste = useCopyPaste();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      handleCopyPaste(event);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleCopyPaste]);
  const { isGraphRunning } = useGraphStore();
  return (
    <div className="flex h-full w-full dark:bg-slate-900">
      <div className="relative flex-1">
        <ReactFlow
          nodes={nodes}
          onNodesChange={onNodesChange}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          edges={edges}
          onConnect={onConnect}
          onEdgesChange={onEdgesChange}
          maxZoom={2}
          minZoom={0.1}
          onDragOver={onDragOver}
          onDrop={onDrop}
          nodesDraggable={!isLocked}
          nodesConnectable={!isLocked}
          elementsSelectable={!isLocked}
        >
          <Background />
          <CustomControls setIsLocked={setIsLocked} isLocked={isLocked} />
          <NewControlPanel />
          <BuilderActions />
          {<GraphLoadingBox flowContentLoading={isFlowContentLoading} />}
          {isGraphRunning && <RunningBackground />}
        </ReactFlow>
      </div>
    </div>
  );
};
