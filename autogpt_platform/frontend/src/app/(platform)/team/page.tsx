"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import {
  KanbanIcon,
  Task01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { notFound } from "next/navigation";
import { AllTasksSection } from "./components/AllTasksSection/AllTasksSection";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertTeamCard } from "./components/ExpertTeamCard/ExpertTeamCard";
import { ExpertTeamCardSkeleton } from "./components/ExpertTeamCardSkeleton";
import { NewPodDialog } from "./components/NewPodDialog/NewPodDialog";
import { PodBoard } from "./components/PodBoard/PodBoard";
import { SoulDrawer } from "./components/SoulDrawer/SoulDrawer";
import { TeamHeaderActions } from "./components/TeamHeaderActions";
import { TeamRoster } from "./components/TeamRoster/TeamRoster";
import { TEAM_GRID_CLASS } from "./helpers";
import { useTeamPage } from "./useTeamPage";

const MAIN_CLASS =
  "mx-auto min-h-screen w-full max-w-[1180px] space-y-6 pb-20 pt-8";

const TABS = [
  { value: "overview", label: "Team Overview", icon: UserGroupIcon },
  { value: "pods", label: "Pod board", icon: KanbanIcon },
  { value: "tasks", label: "All tasks", icon: Task01Icon },
] as const;

export default function TeamPage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const {
    hiredExperts,
    pods,
    podForExpert,
    podGroups,
    ungroupedExperts,
    schedulesForExpert,
    isLoading,
    isError,
    refetch,
    installWorkflow,
    pickerExpertId,
    closeWorkflowPicker,
    soulExpert,
    soulDrawerKey,
    openSoul,
    closeSoul,
    isNewPodOpen,
    openNewPod,
    closeNewPod,
    createPod,
    isCreatingPod,
    assignPod,
  } = useTeamPage({ enabled: Boolean(enabled) && ready });

  if (!ready) {
    return (
      <main className={MAIN_CLASS}>
        <div className={TEAM_GRID_CLASS}>
          {[0, 1, 2].map((i) => (
            <ExpertTeamCardSkeleton key={i} />
          ))}
        </div>
      </main>
    );
  }

  if (!enabled) {
    notFound();
  }

  function renderCard(expert: Expert) {
    return (
      <ExpertTeamCard
        key={expert.id}
        expert={expert}
        schedules={schedulesForExpert(expert)}
        pods={pods}
        currentPod={podForExpert(expert)}
        onInstallWorkflow={installWorkflow}
        onEditSoul={openSoul}
        onAssignPod={assignPod}
      />
    );
  }

  return (
    <main className={MAIN_CLASS}>
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <Icon icon={UserGroupIcon} size={22} className="text-zinc-950" />
            <Text variant="h4">Team</Text>
          </div>
          <Text variant="body" className="max-w-prose text-zinc-600">
            Autopilot and your hired experts, ready to work.
          </Text>
        </div>
        <TeamHeaderActions onNewPod={openNewPod} />
      </div>

      {isError ? (
        <ErrorCard
          context="your team"
          hint="We could not load your hired experts."
          onRetry={() => refetch()}
        />
      ) : null}

      <TabsLine defaultValue="overview">
        <TabsLineList
          flush
          className="overflow-x-auto"
          indicatorClassName="bg-zinc-900"
        >
          {TABS.map((tab) => (
            <TabsLineTrigger
              key={tab.value}
              value={tab.value}
              className="gap-2 data-[state=active]:text-zinc-900"
            >
              <Icon icon={tab.icon} size={16} />
              {tab.label}
            </TabsLineTrigger>
          ))}
        </TabsLineList>

        <TabsLineContent value="overview" className="space-y-6">
          <TeamRoster
            isLoading={isLoading}
            experts={hiredExperts}
            schedulesForExpert={schedulesForExpert}
            renderCard={renderCard}
          />

          {!isLoading && !isError && hiredExperts.length === 0 ? (
            <EmptyTeamState />
          ) : null}
        </TabsLineContent>

        <TabsLineContent value="pods">
          <PodBoard
            isLoading={isLoading}
            podGroups={podGroups}
            ungroupedExperts={ungroupedExperts}
            onNewPod={openNewPod}
          />
        </TabsLineContent>

        <TabsLineContent value="tasks">
          <AllTasksSection
            experts={hiredExperts}
            enabled={Boolean(enabled) && ready}
          />
        </TabsLineContent>
      </TabsLine>

      <InstallWorkflowPicker
        mode="pick-workflow"
        expertId={pickerExpertId ?? undefined}
        open={pickerExpertId !== null}
        onClose={closeWorkflowPicker}
      />
      <SoulDrawer key={soulDrawerKey} expert={soulExpert} onClose={closeSoul} />
      <NewPodDialog
        open={isNewPodOpen}
        onClose={closeNewPod}
        onCreate={createPod}
        isCreating={isCreatingPod}
      />
    </main>
  );
}
