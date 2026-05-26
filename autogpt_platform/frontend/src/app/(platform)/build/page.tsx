"use client";
import { BuilderMobileWarning } from "@/components/layout/BuilderMobileWarning/BuilderMobileWarning";
import { ReactFlowProvider } from "@xyflow/react";
import { Flow } from "./components/FlowEditor/Flow/Flow";

export default function BuilderPage() {
  return (
    <div className="relative h-full w-full">
      <BuilderMobileWarning />
      <ReactFlowProvider>
        <Flow />
      </ReactFlowProvider>
    </div>
  );
}
