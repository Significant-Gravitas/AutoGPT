import { ReactFlow, Background, Controls } from "@xyflow/react";
import { useNodeStore } from "../store/nodeStore";
import { CustomNode } from "./CustomNode/CustomNode";
import { Button } from "@/components/atoms/Button/Button";

export const Flow = () => {
  const initialNodes: CustomNode[] = [
    {
      id: "n1",
      position: { x: 0, y: 0 },
      data: {
        hardcodedValues: {},
        title: "Node 1",
        description: "First node",
        inputSchema: {
          type: "object",
          properties: {
            name: { type: "string" },
          },
          name: "Node 1",
        },
        outputSchema: {
          type: "object",
          properties: {
            name: { type: "string" },
          },
        },
      },
      type: "custom",
    },
    {
      id: "n2",
      position: { x: 100, y: 100 },
      data: {
        hardcodedValues: {},
        title: "Node 2",
        description: "Second node",
        inputSchema: {
          type: "object",
          properties: {
            name: { type: "string" },
          },
        },
        outputSchema: {
          type: "object",
          properties: {
            name: { type: "string" },
          },
        },
      },
      type: "custom",
    },
  ];

  // All these 3 are working perfectly
  const nodes = useNodeStore((state) => state.nodes);
  const onNodesChange = useNodeStore((state) => state.onNodesChange);
  const addNode = useNodeStore((state) => state.addNode);

  const handleAddNode = () => {
    addNode(initialNodes[0]);
  };

  return (
    <div className="h-full w-full bg-slate-900">
      <ReactFlow
        nodes={nodes}
        onNodesChange={onNodesChange}
        nodeTypes={{ custom: CustomNode }}
      >
        <Background />
        <Controls />
        <Button
          className="absolute right-4 top-12 z-50"
          onClick={handleAddNode}
        >
          Add Node
        </Button>
      </ReactFlow>
    </div>
  );
};
