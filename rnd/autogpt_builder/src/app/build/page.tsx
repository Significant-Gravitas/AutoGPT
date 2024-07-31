"use client";

import { useSearchParams } from "next/navigation";
import FlowEditor from '@/components/Flow';

export default function Home() {
  const query = useSearchParams();

  return (
      <FlowEditor
        className="flow-container w-full min-h-[86vh] border border-gray-300 dark:border-gray-700 rounded-lg bg-secondary"
        flowID={query.get("flowID") ?? query.get("templateID") ?? undefined}
        template={!!query.get("templateID")}
      />
  );
}
