"use client";

import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import FlowEditor from "@/app/(platform)/build/components/legacy-builder/Flow/Flow";
// import LoadingBox from "@/components/__legacy__/ui/loading";
import { GraphID } from "@/lib/autogpt-server-api/types";
import { useSearchParams } from "next/navigation";
import { useEffect } from "react";
import { Flow } from "./components/FlowEditor/Flow";
import { BuilderViewTabs } from "./components/BuilderViewTabs/BuilderViewTabs";
import { useBuilderView } from "./components/BuilderViewTabs/useBuilderViewTabs";

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
        {selectedView === "new" ? <Flow /> : <BuilderContent />}
      </div>
    );
  }

  return isNewFlowEditorEnabled ? <Flow /> : <BuilderContent />;
}
