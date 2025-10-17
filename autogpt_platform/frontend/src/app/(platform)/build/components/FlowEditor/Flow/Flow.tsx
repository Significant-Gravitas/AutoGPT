import { ReactFlow, Background, Controls } from "@xyflow/react";
import NewControlPanel from "../../NewControlPanel/NewControlPanel";
import CustomEdge from "../edges/CustomEdge";
import { useFlow } from "./useFlow";
import { useShallow } from "zustand/react/shallow";
import { useNodeStore } from "../../../stores/nodeStore";
import { useMemo } from "react";
import { CustomNode } from "../nodes/CustomNode/CustomNode";
import { useCustomEdge } from "../edges/useCustomEdge";
import { GraphLoadingBox } from "./GraphLoadingBox";

export const Flow = () => {
  const nodes = useNodeStore(useShallow((state) => state.nodes));
  const onNodesChange = useNodeStore(
    useShallow((state) => state.onNodesChange),
  );
  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);
  const { edges, onConnect, onEdgesChange } = useCustomEdge();

  // We use this hook to load the graph and convert them into custom nodes and edges.
  const { isFlowContentLoading } = useFlow();

  return (
    <div className="flex h-full w-full dark:bg-slate-900">
      <div className="relative flex-1">
        <ReactFlow
          nodes={nodes}
          onNodesChange={onNodesChange}
          nodeTypes={nodeTypes}
          edges={edges}
          onConnect={onConnect}
          onEdgesChange={onEdgesChange}
          edgeTypes={{ custom: CustomEdge }}
          maxZoom={2}
          minZoom={0.1}
        >
          <Background />
          <Controls />
          <NewControlPanel />
          {isFlowContentLoading && <GraphLoadingBox />}
        </ReactFlow>
      </div>
    </div>
  );
};
