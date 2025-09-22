"use client";

import FlowEditor from "@/components/Flow";
import LoadingBox from "@/components/ui/loading";
import { GraphID } from "@/lib/autogpt-server-api/types";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

function BuilderContent() {
  const query = useSearchParams();

  const _graphVersion = query.get("flowVersion");
  const graphVersion = _graphVersion ? parseInt(_graphVersion) : undefined;
  return (
    <FlowEditor
      className="flow-container"
      flowID={(query.get("flowID") as GraphID | null) ?? undefined}
      flowVersion={graphVersion}
    />
  );
}

export default function BuilderPage() {
  return (
    <Suspense fallback={<LoadingBox className="h-[80vh]" />}>
      <BuilderContent />
    </Suspense>
  );
}
