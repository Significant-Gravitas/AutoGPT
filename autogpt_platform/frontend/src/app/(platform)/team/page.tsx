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
import { cn } from "@/lib/utils";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { KanbanIcon, UserGroupIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { notFound } from "next/navigation";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertChatDrawer } from "./components/ExpertChatDrawer/ExpertChatDrawer";
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
  "mx-auto min-h-screen w-full max-w-[1180px] space-y-5 px-4 pb-16 pt-6 duration-500 sm:px-8 md:px-12 animate-in fade-in slide-in-from-bottom-2 fill-mode-both motion-reduce:animate-none";

const TABS = [
  { value: "overview", label: "Team Overview", icon: UserGroupIcon },
  { value: "pods", label: "Pod board", icon: KanbanIcon },
] as const;

export default function TeamPage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const {
    hiredExperts,
    pods,
    podForExpert,
    podGroups,
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
    chatTarget,
    chatDrawerKey,
    openChat,
    closeChat,
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
        onChat={openChat}
        onAssignPod={assignPod}
      />
    );
  }

  return (
    <div className="flex w-full">
      <main className={cn(MAIN_CLASS, "min-w-0 flex-1")}>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <Icon icon={UserGroupIcon} size={18} className="text-zinc-950" />
              <Text variant="large-medium" as="h5" tone="primary">
                Team
              </Text>
            </div>
            <Text variant="body" tone="secondary" className="max-w-prose">
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

        <TabsLine variant="compact" defaultValue="overview">
          <TabsLineList className="overflow-x-auto border-b-transparent">
            {TABS.map((tab) => (
              <TabsLineTrigger
                key={tab.value}
                value={tab.value}
                icon={tab.icon}
              >
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
              onAutopilotChat={() => openChat(null)}
            />

            {!isLoading && !isError && hiredExperts.length === 0 ? (
              <EmptyTeamState />
            ) : null}
          </TabsLineContent>

          <TabsLineContent value="pods">
            <PodBoard
              isLoading={isLoading}
              podGroups={podGroups}
              onNewPod={openNewPod}
              renderCard={renderCard}
            />
          </TabsLineContent>
        </TabsLine>

        <InstallWorkflowPicker
          mode="pick-workflow"
          expertId={pickerExpertId ?? undefined}
          open={pickerExpertId !== null}
          onClose={closeWorkflowPicker}
        />
        <NewPodDialog
          open={isNewPodOpen}
          onClose={closeNewPod}
          onCreate={createPod}
          isCreating={isCreatingPod}
        />
      </main>

      <SoulDrawer key={soulDrawerKey} expert={soulExpert} onClose={closeSoul} />
      <ExpertChatDrawer
        target={chatTarget}
        threadKey={chatDrawerKey}
        onClose={closeChat}
        resumeLatest={false}
      />
    </div>
  );
}
