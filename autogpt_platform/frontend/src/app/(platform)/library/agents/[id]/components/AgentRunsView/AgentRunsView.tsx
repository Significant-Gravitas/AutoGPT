"use client";

import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAgentRunsView } from "./useAgentRunsView";
import { AgentRunsLoading } from "./components/AgentRunsLoading";
import { RunsSidebar } from "./components/RunsSidebar/RunsSidebar";
import React from "react";
import { RunDetails } from "./components/RunDetails/RunDetails";

export function AgentRunsView() {
  const { response, ready, error, agentId, selectedRun, handleSelectRun } =
    useAgentRunsView();

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
    <div className="grid h-screen grid-cols-[25%_70%] gap-4 pt-8">
      <RunsSidebar
        agent={agent}
        selectedRunId={selectedRun}
        onSelectRun={handleSelectRun}
      />

      {/* Main Content - 70% */}
      <div className="p-4">
        <Breadcrumbs
          items={[
            { name: "My Library", link: "/library" },
            { name: agent.name, link: `/library/agents/${agentId}` },
          ]}
        />
        <div className="mt-2">
          {selectedRun ? (
            <RunDetails agent={agent} runId={selectedRun} />
          ) : (
            <div className="text-gray-600">
              Select a run to view its details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
