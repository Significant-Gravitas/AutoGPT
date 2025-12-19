"use client";

import { Button } from "@/components/atoms/Button/Button";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { PlusIcon } from "@phosphor-icons/react";
import * as React from "react";
import { RunAgentModal } from "./components/modals/RunAgentModal/RunAgentModal";
import { useMarketplaceUpdate } from "./hooks/useMarketplaceUpdate";
import { AgentVersionChangelog } from "./components/AgentVersionChangelog";
import { MarketplaceBanners } from "../../../../../components/MarketplaceBanners/MarketplaceBanners";
import { AgentRunsLoading } from "./components/other/AgentRunsLoading";
import { EmptySchedules } from "./components/other/EmptySchedules";
import { EmptyTasks } from "./components/other/EmptyTasks";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import { EmptyTemplates } from "./components/other/EmptyTemplates";
import { EmptyTriggers } from "./components/other/EmptyTriggers";
import { SectionWrap } from "./components/other/SectionWrap";
import { LoadingSelectedContent } from "./components/selected-views/LoadingSelectedContent";
import { SelectedRunView } from "./components/selected-views/SelectedRunView/SelectedRunView";
import { SelectedScheduleView } from "./components/selected-views/SelectedScheduleView/SelectedScheduleView";
import { SelectedTemplateView } from "./components/selected-views/SelectedTemplateView/SelectedTemplateView";
import { SelectedTriggerView } from "./components/selected-views/SelectedTriggerView/SelectedTriggerView";
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
    onTriggerSetup,
    onScheduleCreated,
  } = useNewAgentLibraryView();

  const {
    hasAgentMarketplaceUpdate,
    hasMarketplaceUpdate,
    latestMarketplaceVersion,
    isUpdating,
    modalOpen,
    setModalOpen,
    handlePublishUpdate,
    handleUpdateToLatest,
  } = useMarketplaceUpdate({ agent });

  const [changelogOpen, setChangelogOpen] = React.useState(false);

  function renderMarketplaceUpdateBanner() {
    return (
      <MarketplaceBanners
        hasUpdate={!!hasMarketplaceUpdate}
        latestVersion={latestMarketplaceVersion}
        hasUnpublishedChanges={!!hasAgentMarketplaceUpdate}
        currentVersion={agent?.graph_version}
        isUpdating={isUpdating}
        onUpdate={handleUpdateToLatest}
        onPublish={handlePublishUpdate}
        onViewChanges={() => setChangelogOpen(true)}
      />
    );
  }

  function renderPublishAgentModal() {
    if (!modalOpen || !agent) return null;

    return (
      <PublishAgentModal
        targetState={{
          isOpen: true,
          step: "info",
          submissionData: { isMarketplaceUpdate: true } as any,
        }}
        preSelectedAgentId={agent.graph_id}
        preSelectedAgentVersion={agent.graph_version}
        onStateChange={(state) => {
          if (!state.isOpen) {
            setModalOpen(false);
          }
        }}
      />
    );
  }

  function renderAgentLibraryView() {
    if (!agent) return null;

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
          activeTab === "scheduled" ? (
            <SelectedScheduleView
              agent={agent}
              scheduleId={activeItem}
              onClearSelectedRun={handleClearSelectedRun}
              banner={renderMarketplaceUpdateBanner()}
            />
          ) : activeTab === "templates" ? (
            <SelectedTemplateView
              agent={agent}
              templateId={activeItem}
              onClearSelectedRun={handleClearSelectedRun}
              onRunCreated={(execution) =>
                handleSelectRun(execution.id, "runs")
              }
              onSwitchToRunsTab={() => setActiveTab("runs")}
              banner={renderMarketplaceUpdateBanner()}
            />
          ) : activeTab === "triggers" ? (
            <SelectedTriggerView
              agent={agent}
              triggerId={activeItem}
              onClearSelectedRun={handleClearSelectedRun}
              onSwitchToRunsTab={() => setActiveTab("runs")}
              banner={renderMarketplaceUpdateBanner()}
            />
          ) : (
            <SelectedRunView
              agent={agent}
              runId={activeItem}
              onSelectRun={handleSelectRun}
              onClearSelectedRun={handleClearSelectedRun}
              banner={renderMarketplaceUpdateBanner()}
            />
          )
        ) : sidebarLoading ? (
          <LoadingSelectedContent agentName={agent.name} agentId={agent.id} />
        ) : activeTab === "scheduled" ? (
          <SelectedViewLayout
            agentName={agent.name}
            agentId={agent.id}
            banner={renderMarketplaceUpdateBanner()}
          >
            <EmptySchedules />
          </SelectedViewLayout>
        ) : activeTab === "templates" ? (
          <SelectedViewLayout
            agentName={agent.name}
            agentId={agent.id}
            banner={renderMarketplaceUpdateBanner()}
          >
            <EmptyTemplates />
          </SelectedViewLayout>
        ) : activeTab === "triggers" ? (
          <SelectedViewLayout
            agentName={agent.name}
            agentId={agent.id}
            banner={renderMarketplaceUpdateBanner()}
          >
            <EmptyTriggers />
          </SelectedViewLayout>
        ) : (
          <SelectedViewLayout
            agentName={agent.name}
            agentId={agent.id}
            banner={renderMarketplaceUpdateBanner()}
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

  function renderVersionChangelog() {
    if (!agent) return null;

    return (
      <AgentVersionChangelog
        agent={agent}
        isOpen={changelogOpen}
        onClose={() => setChangelogOpen(false)}
      />
    );
  }

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
      <>
        <SelectedViewLayout
          agentName={agent.name}
          agentId={agent.id}
          banner={renderMarketplaceUpdateBanner()}
        >
          <EmptyTasks
            agent={agent}
            onRun={onRunInitiated}
            onTriggerSetup={onTriggerSetup}
            onScheduleCreated={onScheduleCreated}
          />
        </SelectedViewLayout>
        {renderPublishAgentModal()}
        {renderVersionChangelog()}
      </>
    );
  }

  return (
    <>
      {renderAgentLibraryView()}
      {renderPublishAgentModal()}
      {renderVersionChangelog()}
    </>
  );
}
