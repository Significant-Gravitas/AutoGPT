import AgentFlowListSkeleton from "@/app/(platform)/monitoring/components/skeletons/AgentFlowListSkeleton";
import React from "react";
import FlowRunsListSkeleton from "@/app/(platform)/monitoring/components/skeletons/FlowRunsListSkeleton";
import FlowRunsStatusSkeleton from "@/app/(platform)/monitoring/components/skeletons/FlowRunsStatusSkeleton";

export default function MonitorLoadingSkeleton() {
  return (
    <div className="space-y-4 p-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {/* Agents Section */}
        <AgentFlowListSkeleton />

        {/* Runs Section */}
        <FlowRunsListSkeleton />

        {/* Stats Section */}
        <FlowRunsStatusSkeleton />
      </div>
    </div>
  );
}
