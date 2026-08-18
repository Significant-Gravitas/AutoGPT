"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import { cn } from "@/lib/utils";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { notFound } from "next/navigation";
import { CreateMenu } from "./components/CreateMenu/CreateMenu";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertTeamCard } from "./components/ExpertTeamCard/ExpertTeamCard";
import { ExpertTeamCardSkeleton } from "./components/ExpertTeamCardSkeleton";
import { NewPodDialog } from "./components/NewPodDialog/NewPodDialog";
import { SoulDrawer } from "./components/SoulDrawer/SoulDrawer";
import { TeamRoster } from "./components/TeamRoster/TeamRoster";
import { WhatRunsZone } from "./components/WhatRunsZone/WhatRunsZone";
import { SECTION_INSET_CLASS, TEAM_GRID_CLASS } from "./helpers";
import { useTeamPage } from "./useTeamPage";

const MAIN_CLASS =
  "container min-h-screen space-y-6 pb-20 pt-8 sm:px-8 md:px-12";

export default function TeamPage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const {
    hiredExperts,
    pods,
    podForExpert,
    podGroups,
    ungroupedExperts,
    schedules,
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
      <div
        className={cn(
          "flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between",
          SECTION_INSET_CLASS,
        )}
      >
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2.5">
            <AITeamIcon size={36} className="shrink-0 text-zinc-950" />
            <Text variant="h3">Your Team</Text>
          </div>
          <Text variant="body" className="max-w-prose text-zinc-600">
            Autopilot and your hired experts, ready to work.
          </Text>
        </div>
        <CreateMenu onNewPod={openNewPod} />
      </div>
      <TeamRoster
        isLoading={isLoading}
        podGroups={podGroups}
        ungroupedExperts={ungroupedExperts}
        renderCard={renderCard}
      />

      {!isLoading && !isError && hiredExperts.length > 0 ? (
        <WhatRunsZone experts={hiredExperts} schedules={schedules} />
      ) : null}

      {isError ? (
        <ErrorCard
          context="your team"
          hint="We could not load your hired experts."
          onRetry={() => refetch()}
        />
      ) : null}
      {!isLoading &&
      !isError &&
      hiredExperts.length === 0 &&
      podGroups.length === 0 ? (
        <EmptyTeamState />
      ) : null}
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
