import { ReactFlow, Background } from "@xyflow/react";
import NewControlPanel from "../../NewControlPanel/NewControlPanel";
import CustomEdge from "../edges/CustomEdge";
import { useFlow } from "./useFlow";
import { useShallow } from "zustand/react/shallow";
import { useNodeStore } from "../../../stores/nodeStore";
import { useMemo, useEffect, useCallback } from "react";
import { CustomNode } from "../nodes/CustomNode/CustomNode";
import { useCustomEdge } from "../edges/useCustomEdge";
import { useFlowRealtime } from "./useFlowRealtime";
import { GraphLoadingBox } from "./components/GraphLoadingBox";
import { BuilderActions } from "../../BuilderActions/BuilderActions";
import { RunningBackground } from "./components/RunningBackground";
import { useGraphStore } from "../../../stores/graphStore";
import { useCopyPaste } from "./useCopyPaste";
import { FloatingReviewsPanel } from "@/components/organisms/FloatingReviewsPanel/FloatingReviewsPanel";
import { parseAsString, useQueryStates } from "nuqs";
import { CustomControls } from "./components/CustomControl";
import { FloatingSafeModeToggle } from "@/components/molecules/FloatingSafeModeToggle/FloatingSafeModeToggle";
import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { okData } from "@/app/api/helpers";
import { TriggerAgentBanner } from "./components/TriggerAgentBanner";
import { resolveCollisions } from "./helpers/resolve-collision";

export const Flow = () => {
  const [{ flowID, flowExecutionID }] = useQueryStates({
    flowID: parseAsString,
    flowExecutionID: parseAsString,
  });

  const { data: graph } = useGetV1GetSpecificGraph(
    flowID ?? "",
    {},
    {
      query: {
        select: okData<GraphModel>,
        enabled: !!flowID,
      },
    },
  );

  const nodes = useNodeStore(useShallow((state) => state.nodes));
  const setNodes = useNodeStore(useShallow((state) => state.setNodes));
  const onNodesChange = useNodeStore(
    useShallow((state) => state.onNodesChange),
  );
  const hasWebhookNodes = useNodeStore(
    useShallow((state) => state.hasWebhookNodes()),
  );
  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);
  const edgeTypes = useMemo(() => ({ custom: CustomEdge }), []);
  const onNodeDragStop = useCallback(() => {
    setNodes(
      resolveCollisions(nodes, {
        maxIterations: Infinity,
        overlapThreshold: 0.5,
        margin: 15,
      }),
    );
  }, [setNodes, nodes]);
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
  const isGraphRunning = useGraphStore(
    useShallow((state) => state.isGraphRunning),
  );
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
          onNodeDragStop={onNodeDragStop}
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
          {hasWebhookNodes ? <TriggerAgentBanner /> : <BuilderActions />}
          {<GraphLoadingBox flowContentLoading={isFlowContentLoading} />}
          {isGraphRunning && <RunningBackground />}
          {graph && (
            <FloatingSafeModeToggle
              graph={graph}
              className="right-4 top-32 p-2"
              variant="black"
            />
          )}
        </ReactFlow>
      </div>
      {/* TODO: Need to update it in future - also do not send executionId as prop - rather use useQueryState inside the component */}
      <FloatingReviewsPanel
        executionId={flowExecutionID || undefined}
        graphId={flowID || undefined}
      />
    </div>
  );
};
