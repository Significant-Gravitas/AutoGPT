"use client";

import { useSearchParams } from "next/navigation";
import { GraphID } from "@/lib/autogpt-server-api/types";
import FlowEditor from "@/components/Flow";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import { useEffect } from "react";

export default function BuilderPage() {
  const query = useSearchParams();
  const { completeStep } = useOnboarding();

  useEffect(() => {
    completeStep("BUILDER_OPEN");
  }, [completeStep]);

  return (
    <FlowEditor
      className="flow-container"
      flowID={query.get("flowID") as GraphID | null ?? undefined}
      flowVersion={query.get("flowVersion") ?? undefined}
    />
  );
}
