import React from "react";
import { Skeleton } from "@/components/ui/skeleton";

export function AgentRunsLoading() {
  return (
    <div className="px-6 py-6">
      <div className="flex h-screen w-full gap-4">
        {/* Left Sidebar */}
        <div className="w-80 space-y-4">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>

        {/* Main Content */}
        <div className="flex-1 space-y-4">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      </div>
    </div>
  );
}
