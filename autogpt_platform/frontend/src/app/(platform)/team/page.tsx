"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { notFound } from "next/navigation";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertTeamCard } from "./components/ExpertTeamCard/ExpertTeamCard";
import { NewPodDialog } from "./components/NewPodDialog/NewPodDialog";
import { SoulDrawer } from "./components/SoulDrawer/SoulDrawer";
import { TeamRoster } from "./components/TeamRoster/TeamRoster";
import { useTeamPage } from "./useTeamPage";

const MAIN_CLASS =
  "container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12";
const GRID_CLASS = "grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3";

export default function TeamPage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const {
    hiredExperts,
    pods,
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
        <div className={GRID_CLASS}>
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-48 w-full rounded-2xl" />
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
          <div className="flex items-center gap-2.5">
            <AITeamIcon size={36} className="shrink-0 text-black" />
            <Text variant="h3">Your Team</Text>
          </div>
          <Text variant="body" className="max-w-prose text-zinc-600">
            Autopilot and your hired experts, ready to work.
          </Text>
        </div>
        <Button
          size="small"
          leftIcon={<Icon icon={PlusSignIcon} size={16} />}
          onClick={openNewPod}
        >
          New pod
        </Button>
      </div>

      <TeamRoster
        isLoading={isLoading}
        podGroups={podGroups}
        ungroupedExperts={ungroupedExperts}
        renderCard={renderCard}
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
