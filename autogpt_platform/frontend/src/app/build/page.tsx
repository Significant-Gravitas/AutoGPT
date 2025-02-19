"use client";

import { useSearchParams } from "next/navigation";
import { GraphID } from "@/lib/autogpt-server-api/types";
import FlowEditor from "@/components/Flow";

export default function Home() {
  const query = useSearchParams();

  return (
    <FlowEditor
      className="flow-container"
      flowID={query.get("flowID") as GraphID | null ?? undefined}
      flowVersion={query.get("flowVersion") ?? undefined}
    />
  );
}
