"use client";
import { ReactFlowProvider } from "@xyflow/react";
import { Flow } from "./components/FlowEditor/Flow/Flow";

export default function BuilderPage() {
  return (
    <div className="relative h-full w-full">
      <ReactFlowProvider>
        <Flow />
      </ReactFlowProvider>
    </div>
  );
}
