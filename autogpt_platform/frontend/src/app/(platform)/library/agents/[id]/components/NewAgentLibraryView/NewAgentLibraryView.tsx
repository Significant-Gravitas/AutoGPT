"use client";

import { Button } from "@/components/atoms/Button/Button";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { PlusIcon } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { AgentVersionChangelog } from "./components/AgentVersionChangelog";
import { AgentSettingsModal } from "./components/modals/AgentSettingsModal/AgentSettingsModal";
import { RunAgentModal } from "./components/modals/RunAgentModal/RunAgentModal";
import { AgentRunsLoading } from "./components/other/AgentRunsLoading";
import { EmptySchedules } from "./components/other/EmptySchedules";
import { EmptyTasks } from "./components/other/EmptyTasks";
import { EmptyTemplates } from "./components/other/EmptyTemplates";
import { EmptyTriggers } from "./components/other/EmptyTriggers";
import { MarketplaceBanners } from "./components/other/MarketplaceBanners";
import { SectionWrap } from "./components/other/SectionWrap";
import { TriggerNotFound } from "./components/other/TriggerNotFound";
import { LoadingSelectedContent } from "./components/selected-views/LoadingSelectedContent";
import { SelectedRunView } from "./components/selected-views/SelectedRunView/SelectedRunView";
import { SelectedScheduleView } from "./components/selected-views/SelectedScheduleView/SelectedScheduleView";
import { SelectedTemplateView } from "./components/selected-views/SelectedTemplateView/SelectedTemplateView";
import { SelectedTriggerAgentView } from "./components/selected-views/SelectedTriggerAgentView/SelectedTriggerAgentView";
import { SelectedTriggerView } from "./components/selected-views/SelectedTriggerView/SelectedTriggerView";
import { SelectedViewLayout } from "./components/selected-views/SelectedViewLayout";
import { SidebarRunsList } from "./components/sidebar/SidebarRunsList/SidebarRunsList";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "./helpers";
import { useMarketplaceUpdate } from "./hooks/useMarketplaceUpdate";
import { useNewAgentLibraryView } from "./useNewAgentLibraryView";

export function NewAgentLibraryView() {
  const {
    agentId,
    agent,
    ready,
    activeTemplate,
    isTemplateLoading,
    error,
    hasAnyItems,
    activeItemId,
    selectedTriggerKind,
    retryTriggerLists,
    sidebarLoading,
    activeTab,
    setActiveTab,
    handleSelectRun,
    handleCountsChange,
    handleClearSelectedRun,
    handleScheduleDeleted,
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

  const [changelogOpen, setChangelogOpen] = useState(false);

  useEffect(() => {
    if (agent) {
      document.title = `${agent.name} - Library - AutoGPT Platform`;
    }
  }, [agent]);

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

  function renderSelectedTrigger(selectedItemId: string) {
    if (!agent) return null;

    switch (selectedTriggerKind) {
      case "trigger-agent":
        return (
          <SelectedTriggerAgentView
            agent={agent}
            triggerAgentId={selectedItemId}
            onClearSelectedRun={handleClearSelectedRun}
            banner={renderMarketplaceUpdateBanner()}
          />
        );
      case "webhook-trigger":
        return (
          <SelectedTriggerView
            agent={agent}
            triggerId={selectedItemId}
            onClearSelectedRun={handleClearSelectedRun}
            onSwitchToRunsTab={() => setActiveTab("runs")}
            banner={renderMarketplaceUpdateBanner()}
          />
        );
      case "loading":
        return <LoadingSelectedContent agent={agent} />;
      case "error":
        return (
          <SelectedViewLayout
            agent={agent}
            banner={renderMarketplaceUpdateBanner()}
          >
            <ErrorCard
              responseError={{
                message:
                  "Could not load this agent's triggers. Check your connection and try again.",
              }}
              context="triggers"
              onRetry={retryTriggerLists}
            />
          </SelectedViewLayout>
        );
      case "not-found":
        return (
          <TriggerNotFound
            agent={agent}
            banner={renderMarketplaceUpdateBanner()}
            onClearSelection={handleClearSelectedRun}
          />
        );
      default:
        return null;
    }
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

  // Keep the selected-content layout while an item is selected — even with
  // zero listable items — so a stale selection can show its not-found state.
  if (!sidebarLoading && !hasAnyItems && !activeItemId) {
    return (
      <>
        <div className="flex h-full flex-col">
          <div className="mx-6 flex flex-col gap-4 pt-4">
            <div className="flex items-center justify-between">
              <Breadcrumbs
                items={[
                  { name: "My Library", link: "/library" },
                  { name: agent.name, link: `/library/agents/${agentId}` },
                ]}
              />
              <AgentSettingsModal agent={agent} />
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
        {renderPublishAgentModal()}
        {renderVersionChangelog()}
      </>
    );
  }

  return (
    <>
      <div className="mx-4 grid h-full w-full grid-cols-1 gap-0 pt-3 md:ml-4 md:mr-0 md:gap-4 lg:grid-cols-[25%_70%]">
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
                  variant="outline"
                  size="small"
                  className="w-full"
                  disabled={isTemplateLoading && activeTab === "templates"}
                >
                  <PlusIcon size={16} /> New agent task
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
            selectedRunId={activeItemId ?? undefined}
            onSelectRun={handleSelectRun}
            onClearSelectedRun={handleClearSelectedRun}
            onScheduleDeleted={handleScheduleDeleted}
            onTabChange={setActiveTab}
            onCountsChange={handleCountsChange}
          />
        </SectionWrap>

        {activeItemId ? (
          activeTab === "scheduled" ? (
            <SelectedScheduleView
              agent={agent}
              scheduleId={activeItemId}
              onScheduleDeleted={handleScheduleDeleted}
              onSelectRun={(id) => handleSelectRun(id, "runs")}
              banner={renderMarketplaceUpdateBanner()}
            />
          ) : activeTab === "templates" ? (
            <SelectedTemplateView
              agent={agent}
              templateId={activeItemId}
              onClearSelectedRun={handleClearSelectedRun}
              onRunCreated={(execution) =>
                handleSelectRun(execution.id, "runs")
              }
              onSwitchToRunsTab={() => setActiveTab("runs")}
              banner={renderMarketplaceUpdateBanner()}
            />
          ) : activeTab === "triggers" ? (
            renderSelectedTrigger(activeItemId)
          ) : (
            <SelectedRunView
              agent={agent}
              runId={activeItemId}
              onSelectRun={handleSelectRun}
              onClearSelectedRun={handleClearSelectedRun}
              banner={renderMarketplaceUpdateBanner()}
            />
          )
        ) : sidebarLoading ? (
          <LoadingSelectedContent agent={agent} />
        ) : activeTab === "scheduled" ? (
          <SelectedViewLayout
            agent={agent}
            banner={renderMarketplaceUpdateBanner()}
          >
            <EmptySchedules />
          </SelectedViewLayout>
        ) : activeTab === "templates" ? (
          <SelectedViewLayout
            agent={agent}
            banner={renderMarketplaceUpdateBanner()}
          >
            <EmptyTemplates />
          </SelectedViewLayout>
        ) : activeTab === "triggers" ? (
          <SelectedViewLayout
            agent={agent}
            banner={renderMarketplaceUpdateBanner()}
          >
            <EmptyTriggers />
          </SelectedViewLayout>
        ) : (
          <SelectedViewLayout
            agent={agent}
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
      {renderPublishAgentModal()}
      {renderVersionChangelog()}
    </>
  );
}
