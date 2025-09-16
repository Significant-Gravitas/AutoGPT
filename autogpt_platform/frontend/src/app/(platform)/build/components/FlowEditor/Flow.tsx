import { ReactFlow, Background, Controls } from "@xyflow/react";
import { useNodeStore } from "../store/nodeStore";

import NewControlPanel from "../NewBlockMenu/NewControlPanel/NewControlPanel";
import { useShallow } from "zustand/react/shallow";
import { useMemo } from "react";
import { CustomNode } from "./nodes/CustomNode";
import { useCustomEdge } from "./edges/useCustomEdge";
import { RightSidebar } from "../RIghtSidebar";
import CustomEdge from "./edges/CustomEdge";

export const Flow = () => {
  // All these 3 are working perfectly
  const nodes = useNodeStore(useShallow((state) => state.nodes));
  const onNodesChange = useNodeStore(
    useShallow((state) => state.onNodesChange),
  );
  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);
  const { edges, onConnect, onEdgesChange } = useCustomEdge();

  return (
    <div className="realtw-full h-full dark:bg-slate-900">
      <ReactFlow
        nodes={nodes}
        onNodesChange={onNodesChange}
        nodeTypes={nodeTypes}
        edges={edges}
        onConnect={onConnect}
        onEdgesChange={onEdgesChange}
        edgeTypes={{ custom: CustomEdge }}
      >
        <Background />
        <Controls />
        <NewControlPanel />
        <RightSidebar />
      </ReactFlow>
    </div>
  );
};
