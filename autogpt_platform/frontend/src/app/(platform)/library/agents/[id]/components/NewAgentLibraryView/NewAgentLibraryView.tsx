"use client";

import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { PlusIcon } from "@phosphor-icons/react";
import { RunAgentModal } from "./components/modals/RunAgentModal/RunAgentModal";
import { AgentRunsLoading } from "./components/other/AgentRunsLoading";
import { EmptySchedules } from "./components/other/EmptySchedules";
import { EmptyTasks } from "./components/other/EmptyTasks";
import { EmptyTemplates } from "./components/other/EmptyTemplates";
import { SectionWrap } from "./components/other/SectionWrap";
import { SelectedRunView } from "./components/selected-views/SelectedRunView/SelectedRunView";
import { SelectedScheduleView } from "./components/selected-views/SelectedScheduleView/SelectedScheduleView";
import { SidebarRunsList } from "./components/sidebar/SidebarRunsList/SidebarRunsList";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "./helpers";
import { useNewAgentLibraryView } from "./useNewAgentLibraryView";

export function NewAgentLibraryView() {
  const {
    agent,
    hasAnyItems,
    ready,
    error,
    agentId,
    activeItem,
    sidebarLoading,
    activeTab,
    setActiveTab,
    handleSelectRun,
    handleCountsChange,
    handleClearSelectedRun,
  } = useNewAgentLibraryView();

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

  if (!sidebarLoading && !hasAnyItems) {
    return (
      <div className="flex h-full flex-col">
        <div className="mx-6 pt-4">
          <Breadcrumbs
            items={[
              { name: "My Library", link: "/library" },
              { name: agent.name, link: `/library/agents/${agentId}` },
            ]}
          />
        </div>
        <div className="flex min-h-0 flex-1">
          <EmptyTasks agent={agent} />
        </div>
      </div>
    );
  }

  return (
    <div className="ml-4 grid h-full grid-cols-1 gap-0 pt-3 md:gap-4 lg:grid-cols-[25%_70%]">
      <SectionWrap className="mb-3 block">
        <div
          className={cn(
            "border-b border-zinc-100 pb-5",
            AGENT_LIBRARY_SECTION_PADDING_X,
          )}
        >
          <RunAgentModal
            triggerSlot={
              <Button variant="primary" size="large" className="w-full">
                <PlusIcon size={20} /> New task
              </Button>
            }
            agent={agent}
            agentId={agent.id.toString()}
            onRunCreated={(execution) => handleSelectRun(execution.id, "runs")}
            onScheduleCreated={(schedule) =>
              handleSelectRun(schedule.id, "scheduled")
            }
          />
        </div>

        <SidebarRunsList
          agent={agent}
          selectedRunId={activeItem ?? undefined}
          onSelectRun={handleSelectRun}
          onClearSelectedRun={handleClearSelectedRun}
          onTabChange={setActiveTab}
          onCountsChange={handleCountsChange}
        />
      </SectionWrap>

      <SectionWrap className="mb-3">
        <div
          className={`${AGENT_LIBRARY_SECTION_PADDING_X} border-b border-zinc-100 pb-4`}
        >
          <Breadcrumbs
            items={[
              { name: "My Library", link: "/library" },
              { name: agent.name, link: `/library/agents/${agentId}` },
            ]}
          />
        </div>
        <div className="flex min-h-0 flex-1 flex-col">
          {activeItem ? (
            activeTab === "scheduled" ? (
              <SelectedScheduleView
                agent={agent}
                scheduleId={activeItem}
                onClearSelectedRun={handleClearSelectedRun}
              />
            ) : (
              <SelectedRunView
                agent={agent}
                runId={activeItem}
                onSelectRun={handleSelectRun}
                onClearSelectedRun={handleClearSelectedRun}
              />
            )
          ) : sidebarLoading ? (
            <div className="flex flex-col gap-4">
              <Skeleton className="h-8 w-full bg-slate-100" />
              <Skeleton className="h-12 w-full bg-slate-100" />
              <Skeleton className="h-64 w-full bg-slate-100" />
              <Skeleton className="h-32 w-full bg-slate-100" />
            </div>
          ) : activeTab === "scheduled" ? (
            <EmptySchedules />
          ) : activeTab === "templates" ? (
            <EmptyTemplates />
          ) : (
            <EmptyTasks agent={agent} />
          )}
        </div>
      </SectionWrap>
    </div>
  );
}
