"use client";
import { ReactFlowProvider } from "@xyflow/react";
import { Flow } from "./components/FlowEditor/Flow/Flow";
import { MobileWarning } from "./components/MobileWarning/MobileWarning";

export default function BuilderPage() {
  return (
    <div className="relative h-full w-full">
      <ReactFlowProvider>
        <Flow />
      </ReactFlowProvider>
      <MobileWarning />
    </div>
  );
}
