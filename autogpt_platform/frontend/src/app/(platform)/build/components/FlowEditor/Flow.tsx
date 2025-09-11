import { ReactFlow, Background, Controls } from "@xyflow/react";
import { useNodeStore } from "../store/nodeStore";

import NewControlPanel from "../NewBlockMenu/NewControlPanel/NewControlPanel";
import { useShallow } from "zustand/react/shallow";
import { useMemo } from "react";
import { CustomNode } from "./CustomNode/CustomNode";

export const Flow = () => {
  // All these 3 are working perfectly
  const nodes = useNodeStore(useShallow((state) => state.nodes));
  const onNodesChange = useNodeStore(
    useShallow((state) => state.onNodesChange),
  );
  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);

  return (
    <div className="realtw-full h-full dark:bg-slate-900">
      <ReactFlow
        nodes={nodes}
        onNodesChange={onNodesChange}
        nodeTypes={nodeTypes}
      >
        <Background />
        <Controls />
        <NewControlPanel />
      </ReactFlow>
    </div>
  );
};
