"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { PlusIcon } from "@phosphor-icons/react";
import { useEffect } from "react";
import { RunAgentModal } from "./components/modals/RunAgentModal/RunAgentModal";
import { AgentRunsLoading } from "./components/other/AgentRunsLoading";
import { EmptyAgentRuns } from "./components/other/EmptyAgentRuns";
import { SelectedRunView } from "./components/selected-views/SelectedRunView/SelectedRunView";
import { SelectedScheduleView } from "./components/selected-views/SelectedScheduleView/SelectedScheduleView";
import { AgentRunsLists } from "./components/sidebar/AgentRunsLists/AgentRunsLists";
import { useNewAgentLibraryView } from "./useNewAgentLibraryView";

export function NewAgentLibraryView() {
  const {
    agent,
    hasAnyItems,
    showSidebarLayout,
    ready,
    error,
    agentId,
    selectedRun,
    sidebarLoading,
    handleSelectRun,
    handleCountsChange,
    handleClearSelectedRun,
  } = useNewAgentLibraryView();

  useEffect(() => {
    if (agent) {
      document.title = `${agent.name} - Library - AutoGPT Platform`;
    }
  }, [agent]);

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

        <AgentRunsLists
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
          ) : sidebarLoading ? (
            // Show loading state while sidebar is loading to prevent flash of empty state
            <div className="text-gray-600">Loading runs...</div>
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
