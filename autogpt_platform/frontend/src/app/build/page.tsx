"use client";

import { useSearchParams } from "next/navigation";
import { GraphID } from "@/lib/autogpt-server-api/types";
import FlowEditor from "@/components/Flow";
import React from "react";
import BuildFlow from "@/components/build/BuildFlow";
import { CustomNode } from "@/components/CustomNode";

export default function Home() {
  const query = useSearchParams();

  const nodeTypes = { custom: CustomNode };

  // Define initial nodes and edges
  const initialNodes = [
    {
      id: "1",
      type: "custom", // Type must match the key in the `nodeTypes` object
      position: { x: 100, y: 100 },
      data: { label: "I am a custom node" }, // Must match `CustomNodeData`
    },
    {
      id: "2",
      type: "custom",
      position: { x: 400, y: 200 },
      data: { label: "Another custom node" },
    },
  ];

  const initialEdges = [
    { id: "e1-2", source: "1", target: "2", type: "default" },
  ];

  return (
    <div className="flow-container">
      <BuildFlow
        id="example-canvas"
        initialNodes={initialNodes}
        initialEdges={initialEdges}
        readOnly={false}
      />
    </div>
  );
}
