"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { PlusIcon } from "@phosphor-icons/react";
import { useEffect } from "react";
import { RunAgentModal } from "./components/modals/RunAgentModal/RunAgentModal";
import { AgentRunsLoading } from "./components/other/AgentRunsLoading";
import { EmptySchedules } from "./components/other/EmptySchedules";
import { EmptyTasks } from "./components/other/EmptyTasks";
import { EmptyTemplates } from "./components/other/EmptyTemplates";
import { EmptyTriggers } from "./components/other/EmptyTriggers";
import { SectionWrap } from "./components/other/SectionWrap";
import { LoadingSelectedContent } from "./components/selected-views/LoadingSelectedContent";
import { SelectedRunView } from "./components/selected-views/SelectedRunView/SelectedRunView";
import { SelectedScheduleView } from "./components/selected-views/SelectedScheduleView/SelectedScheduleView";
import { SelectedTemplateView } from "./components/selected-views/SelectedTemplateView/SelectedTemplateView";
import { SelectedTriggerView } from "./components/selected-views/SelectedTriggerView/SelectedTriggerView";
import { SelectedSettingsView } from "./components/selected-views/SelectedSettingsView/SelectedSettingsView";
import { SelectedViewLayout } from "./components/selected-views/SelectedViewLayout";
import { SidebarRunsList } from "./components/sidebar/SidebarRunsList/SidebarRunsList";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "./helpers";
import { useNewAgentLibraryView } from "./useNewAgentLibraryView";

export function NewAgentLibraryView() {
  const {
    agent,
    ready,
    activeTemplate,
    isTemplateLoading,
    error,
    hasAnyItems,
    activeItem,
    sidebarLoading,
    activeTab,
    setActiveTab,
    handleSelectRun,
    handleCountsChange,
    handleClearSelectedRun,
    onRunInitiated,
    handleSelectSettings,
    onTriggerSetup,
    onScheduleCreated,
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

  if (!sidebarLoading && !hasAnyItems) {
    return (
      <div className="flex h-full flex-col">
        <div className="mx-6 pt-4">
          <div className="relative flex items-center gap-2">
            <Breadcrumbs
              items={[
                { name: "My Library", link: "/library" },
                { name: agent.name },
              ]}
            />
          </div>
        </div>
        <div className="flex min-h-0 flex-1">
          <EmptyTasks
            agent={agent}
            onRun={onRunInitiated}
            onTriggerSetup={onTriggerSetup}
            onScheduleCreated={onScheduleCreated}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="mx-4 grid h-full grid-cols-1 gap-0 pt-3 md:ml-4 md:mr-0 md:gap-4 lg:grid-cols-[25%_70%]">
      <SectionWrap className="mb-3 block">
        <div
          className={cn(
            "border-b border-zinc-100 pb-5",
            AGENT_LIBRARY_SECTION_PADDING_X,
          )}
        >
          <RunAgentModal
            triggerSlot={
              <Button
                variant="primary"
                size="large"
                className="w-full"
                disabled={isTemplateLoading && activeTab === "templates"}
              >
                <PlusIcon size={20} /> New task
              </Button>
            }
            agent={agent}
            onRunCreated={onRunInitiated}
            onScheduleCreated={onScheduleCreated}
            onTriggerSetup={onTriggerSetup}
            initialInputValues={activeTemplate?.inputs}
            initialInputCredentials={activeTemplate?.credentials}
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

      {activeItem ? (
        activeItem === "settings" ? (
          <SelectedSettingsView
            agent={agent}
            onClearSelectedRun={handleClearSelectedRun}
          />
        ) : activeTab === "scheduled" ? (
          <SelectedScheduleView
            agent={agent}
            scheduleId={activeItem}
            onClearSelectedRun={handleClearSelectedRun}
          />
        ) : activeTab === "templates" ? (
          <SelectedTemplateView
            agent={agent}
            templateId={activeItem}
            onClearSelectedRun={handleClearSelectedRun}
            onRunCreated={(execution) => handleSelectRun(execution.id, "runs")}
            onSwitchToRunsTab={() => setActiveTab("runs")}
          />
        ) : activeTab === "triggers" ? (
          <SelectedTriggerView
            agent={agent}
            triggerId={activeItem}
            onClearSelectedRun={handleClearSelectedRun}
            onSwitchToRunsTab={() => setActiveTab("runs")}
          />
        ) : (
          <SelectedRunView
            agent={agent}
            runId={activeItem}
            onSelectRun={handleSelectRun}
            onClearSelectedRun={handleClearSelectedRun}
            onSelectSettings={handleSelectSettings}
          />
        )
      ) : sidebarLoading ? (
        <LoadingSelectedContent
          agent={agent}
          onSelectSettings={handleSelectSettings}
        />
      ) : activeTab === "scheduled" ? (
        <SelectedViewLayout
          agent={agent}
          onSelectSettings={handleSelectSettings}
        >
          <EmptySchedules />
        </SelectedViewLayout>
      ) : activeTab === "templates" ? (
        <SelectedViewLayout
          agent={agent}
          onSelectSettings={handleSelectSettings}
        >
          <EmptyTemplates />
        </SelectedViewLayout>
      ) : activeTab === "triggers" ? (
        <SelectedViewLayout
          agent={agent}
          onSelectSettings={handleSelectSettings}
        >
          <EmptyTriggers />
        </SelectedViewLayout>
      ) : (
        <SelectedViewLayout
          agent={agent}
          onSelectSettings={handleSelectSettings}
        >
          <EmptyTasks
            agent={agent}
            onRun={onRunInitiated}
            onTriggerSetup={onTriggerSetup}
            onScheduleCreated={onScheduleCreated}
          />
        </SelectedViewLayout>
      )}
    </div>
  );
}
