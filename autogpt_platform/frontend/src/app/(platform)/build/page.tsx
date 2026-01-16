"use client";

import FlowEditor from "@/app/(platform)/build/components/legacy-builder/Flow/Flow";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
// import LoadingBox from "@/components/__legacy__/ui/loading";
import { GraphID } from "@/lib/autogpt-server-api/types";
import { ReactFlowProvider } from "@xyflow/react";
import { useSearchParams } from "next/navigation";
import { useEffect } from "react";
import { BuilderViewTabs } from "./components/BuilderViewTabs/BuilderViewTabs";
import { Flow } from "./components/FlowEditor/Flow/Flow";
import { useBuilderView } from "./useBuilderView";

function BuilderContent() {
  const query = useSearchParams();
  const { completeStep } = useOnboarding();

  useEffect(() => {
    completeStep("BUILDER_OPEN");
  }, [completeStep]);

  const _graphVersion = query.get("flowVersion");
  const graphVersion = _graphVersion ? parseInt(_graphVersion) : undefined;
  return (
    <FlowEditor
      className="flex h-full w-full"
      flowID={(query.get("flowID") as GraphID | null) ?? undefined}
      flowVersion={graphVersion}
    />
  );
}

export default function BuilderPage() {
  const {
    isSwitchEnabled,
    selectedView,
    setSelectedView,
    isNewFlowEditorEnabled,
  } = useBuilderView();

  // Switch is temporary, we will remove it once our new flow editor is ready
  if (isSwitchEnabled) {
    return (
      <div className="relative h-full w-full">
        <BuilderViewTabs value={selectedView} onChange={setSelectedView} />
        {selectedView === "new" ? (
          <ReactFlowProvider>
            <Flow />
          </ReactFlowProvider>
        ) : (
          <BuilderContent />
        )}
      </div>
    );
  }

  return isNewFlowEditorEnabled ? (
    <ReactFlowProvider>
      <Flow />
    </ReactFlowProvider>
  ) : (
    <BuilderContent />
  );
}
