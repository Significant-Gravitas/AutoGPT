"use client";

import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAgentRunsView } from "./useAgentRunsView";
import { AgentRunsLoading } from "./components/AgentRunsLoading";
import { RunsSidebar } from "./components/RunsSidebar/RunsSidebar";
import React, { useMemo, useState } from "react";
import { RunDetails } from "./components/RunDetails/RunDetails";
import { ScheduleDetails } from "./components/ScheduleDetails/ScheduleDetails";
import { EmptyAgentRuns } from "./components/EmptyAgentRuns/EmptyAgentRuns";

export function AgentRunsView() {
  const {
    response,
    ready,
    error,
    agentId,
    selectedRun,
    handleSelectRun,
    clearSelectedRun,
  } = useAgentRunsView();
  const [sidebarCounts, setSidebarCounts] = useState({
    runsCount: 0,
    schedulesCount: 0,
  });

  const hasAnyItems = useMemo(
    () =>
      (sidebarCounts.runsCount ?? 0) > 0 ||
      (sidebarCounts.schedulesCount ?? 0) > 0,
    [sidebarCounts],
  );

  if (!ready) {
    return <AgentRunsLoading />;
  }

  if (error) {
    return (
      <ErrorCard
        isSuccess={false}
        responseError={error || undefined}
        httpError={
          response?.status !== 200
            ? {
                status: response?.status,
                statusText: "Request failed",
              }
            : undefined
        }
        context="agent"
        onRetry={() => window.location.reload()}
      />
    );
  }

  if (!response?.data || response.status !== 200) {
    return (
      <ErrorCard
        isSuccess={false}
        responseError={{ message: "No agent data found" }}
        context="agent"
        onRetry={() => window.location.reload()}
      />
    );
  }

  const agent = response.data;

  return (
    <div
      className={
        hasAnyItems
          ? "grid h-screen grid-cols-1 gap-0 pt-6 md:gap-4 lg:grid-cols-[25%_70%]"
          : "grid h-screen grid-cols-1 gap-0 pt-6 md:gap-4"
      }
    >
      <div className={hasAnyItems ? "" : "hidden"}>
        <RunsSidebar
          agent={agent}
          selectedRunId={selectedRun}
          onSelectRun={handleSelectRun}
          onCountsChange={setSidebarCounts}
        />
      </div>

      {/* Main Content - 70% */}
      <div className="p-4">
        <div className={!hasAnyItems ? "px-2" : ""}>
          <Breadcrumbs
            items={[
              { name: "My Library", link: "/library" },
              { name: agent.name, link: `/library/agents/${agentId}` },
            ]}
          />
        </div>
        <div className="mt-1">
          {selectedRun ? (
            selectedRun.startsWith("schedule:") ? (
              <ScheduleDetails
                agent={agent}
                scheduleId={selectedRun.replace("schedule:", "")}
                onClearSelectedRun={clearSelectedRun}
              />
            ) : (
              <RunDetails
                agent={agent}
                runId={selectedRun}
                onSelectRun={handleSelectRun}
                onClearSelectedRun={clearSelectedRun}
              />
            )
          ) : hasAnyItems ? (
            <div className="text-gray-600">
              Select a run to view its details
            </div>
          ) : (
            <EmptyAgentRuns
              agentName={agent.name}
              creatorName={agent.creator_name || "Unknown"}
              description={agent.description}
              agent={agent}
            />
          )}
        </div>
      </div>
    </div>
  );
}
