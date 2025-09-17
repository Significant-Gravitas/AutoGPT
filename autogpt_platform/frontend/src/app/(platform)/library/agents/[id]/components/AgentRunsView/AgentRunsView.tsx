"use client";

import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAgentRunsView } from "./useAgentRunsView";
import { AgentRunsLoading } from "./components/AgentRunsLoading";
import { RunsSidebar } from "./components/RunsSidebar/RunsSidebar";
import { SelectedRunView } from "./components/SelectedRunView/SelectedRunView";
import { SelectedScheduleView } from "./components/SelectedScheduleView/SelectedScheduleView";
import { EmptyAgentRuns } from "./components/EmptyAgentRuns/EmptyAgentRuns";
import { Button } from "@/components/atoms/Button/Button";
import { RunAgentModal } from "./components/RunAgentModal/RunAgentModal";
import { PlusIcon } from "@phosphor-icons/react";

export function AgentRunsView() {
  const {
    agent,
    hasAnyItems,
    showSidebarLayout,
    ready,
    error,
    agentId,
    selectedRun,
    handleSelectRun,
    handleCountsChange,
    handleClearSelectedRun,
  } = useAgentRunsView();

  if (error) {
    return (
      <ErrorCard
        isSuccess={false}
        responseError={error || undefined}
        context="agent"
        onRetry={() => window.location.reload()}
      />
    );
  }

  if (!ready || !agent) {
    return <AgentRunsLoading />;
  }

  return (
    <div
      className={
        showSidebarLayout
          ? "grid h-screen grid-cols-1 gap-0 pt-3 md:gap-4 lg:grid-cols-[25%_70%]"
          : "grid h-screen grid-cols-1 gap-0 pt-3 md:gap-4"
      }
    >
      <div className={showSidebarLayout ? "p-4 pl-5" : "hidden p-4 pl-5"}>
        <div className="mb-4">
          <RunAgentModal
            triggerSlot={
              <Button variant="primary" size="large" className="w-full">
                <PlusIcon size={20} /> New Run
              </Button>
            }
            agent={agent}
            agentId={agent.id.toString()}
            onRunCreated={(execution) => handleSelectRun(execution.id)}
            onScheduleCreated={(schedule) =>
              handleSelectRun(`schedule:${schedule.id}`)
            }
          />
        </div>

        <RunsSidebar
          agent={agent}
          selectedRunId={selectedRun}
          onSelectRun={handleSelectRun}
          onCountsChange={handleCountsChange}
        />
      </div>

      {/* Main Content - 70% */}
      <div className="p-4">
        <div className={!showSidebarLayout ? "px-2" : ""}>
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
              <SelectedScheduleView
                agent={agent}
                scheduleId={selectedRun.replace("schedule:", "")}
                onClearSelectedRun={handleClearSelectedRun}
              />
            ) : (
              <SelectedRunView
                agent={agent}
                runId={selectedRun}
                onSelectRun={handleSelectRun}
                onClearSelectedRun={handleClearSelectedRun}
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
