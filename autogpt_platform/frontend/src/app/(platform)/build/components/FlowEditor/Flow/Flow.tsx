import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { okData } from "@/app/api/helpers";
import { FloatingReviewsPanel } from "@/components/organisms/FloatingReviewsPanel/FloatingReviewsPanel";
import { Background, ReactFlow } from "@xyflow/react";
import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useGraphStore } from "../../../stores/graphStore";
import { useNodeStore } from "../../../stores/nodeStore";
import { BuilderActions } from "../../BuilderActions/BuilderActions";
import { DraftRecoveryPopup } from "../../DraftRecoveryDialog/DraftRecoveryPopup";
import { FloatingSafeModeToggle } from "../../FloatingSafeModeToogle";
import NewControlPanel from "../../NewControlPanel/NewControlPanel";
import CustomEdge from "../edges/CustomEdge";
import { useCustomEdge } from "../edges/useCustomEdge";
import { CustomNode } from "../nodes/CustomNode/CustomNode";
import { CustomControls } from "./components/CustomControl";
import { GraphLoadingBox } from "./components/GraphLoadingBox";
import { RunningBackground } from "./components/RunningBackground";
import { TriggerAgentBanner } from "./components/TriggerAgentBanner";
import { resolveCollisions } from "./helpers/resolve-collision";
import { useCopyPaste } from "./useCopyPaste";
import { useFlow } from "./useFlow";
import { useFlowRealtime } from "./useFlowRealtime";

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
        select: okData,
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
    const currentNodes = useNodeStore.getState().nodes;
    setNodes(
      resolveCollisions(currentNodes, {
        maxIterations: Infinity,
        overlapThreshold: 0.5,
        margin: 15,
      }),
    );
  }, [setNodes]);

  const { edges, onConnect, onEdgesChange } = useCustomEdge();

  // for loading purpose
  const {
    onDragOver,
    onDrop,
    isFlowContentLoading,
    isInitialLoadComplete,
    isLocked,
    setIsLocked,
  } = useFlow();

  // This hook is used for websocket realtime updates.
  useFlowRealtime();

  // Copy/paste functionality
  useCopyPaste();

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
          onNodeContextMenu={(event) => {
            event.preventDefault();
          }}
          maxZoom={2}
          minZoom={0.1}
          onDragOver={onDragOver}
          onDrop={onDrop}
          nodesDraggable={!isLocked}
          nodesConnectable={!isLocked}
          elementsSelectable={!isLocked}
          deleteKeyCode={["Backspace", "Delete"]}
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
              className="right-2 top-32 p-2"
            />
          )}
          <DraftRecoveryPopup isInitialLoadComplete={isInitialLoadComplete} />
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
