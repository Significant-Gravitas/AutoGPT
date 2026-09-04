"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { notFound } from "next/navigation";
import { CreateMenu } from "./components/CreateMenu/CreateMenu";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertRow } from "./components/ExpertRow/ExpertRow";
import { ExpertRowSkeleton } from "./components/ExpertRowSkeleton";
import { NewPodDialog } from "./components/NewPodDialog/NewPodDialog";
import { SoulDrawer } from "./components/SoulDrawer/SoulDrawer";
import { TeamRoster } from "./components/TeamRoster/TeamRoster";
import { WhatRunsZone } from "./components/WhatRunsZone/WhatRunsZone";
import { LIST_SURFACE_CLASS } from "./helpers";
import { useTeamPage } from "./useTeamPage";

const MAIN_CLASS =
  "container flex min-h-screen flex-col gap-10 pb-20 pt-8 sm:px-8 md:px-12";

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
        <div className={LIST_SURFACE_CLASS}>
          {[0, 1, 2].map((i) => (
            <ExpertRowSkeleton key={i} />
          ))}
        </div>
      </main>
    );
  }

  if (!enabled) {
    notFound();
  }

  function renderRow(expert: Expert) {
    return (
      <ExpertRow
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
      <div className="flex flex-col gap-4">
        <header className="flex items-center justify-between gap-4">
          <Text variant="h4" as="h1">
            Team
          </Text>
          <CreateMenu onNewPod={openNewPod} />
        </header>

        <TeamRoster
          isLoading={isLoading}
          podGroups={podGroups}
          ungroupedExperts={ungroupedExperts}
          renderRow={renderRow}
        />

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
      </div>

      {!isLoading && !isError && hiredExperts.length > 0 ? (
        <WhatRunsZone experts={hiredExperts} schedules={schedules} />
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
