"use client";

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
import { Expert } from "@/app/api/__generated__/models/expert";
import { AutopilotCard } from "./components/AutopilotCard";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertTeamCard } from "./components/ExpertTeamCard/ExpertTeamCard";
import { NewPodDialog } from "./components/NewPodDialog/NewPodDialog";
import { SoulDrawer } from "./components/SoulDrawer/SoulDrawer";
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

      {isLoading ? (
        <div className={GRID_CLASS}>
          <AutopilotCard />
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-48 w-full rounded-2xl" />
          ))}
        </div>
      ) : podGroups.length === 0 ? (
        <div className={GRID_CLASS}>
          <AutopilotCard />
          {ungroupedExperts.map(renderCard)}
        </div>
      ) : (
        <div className="space-y-8">
          <div className={GRID_CLASS}>
            <AutopilotCard />
          </div>
          {podGroups.map((group) => (
            <section key={group.pod.id} className="space-y-3">
              <div className="flex items-baseline gap-2">
                <Text variant="h4">{group.pod.name}</Text>
                <Text variant="small" className="text-zinc-500">
                  {group.experts.length}{" "}
                  {group.experts.length === 1 ? "expert" : "experts"}
                </Text>
              </div>
              {group.experts.length > 0 ? (
                <div className={GRID_CLASS}>
                  {group.experts.map(renderCard)}
                </div>
              ) : (
                <Text variant="small" className="text-zinc-500">
                  No experts in this pod yet.
                </Text>
              )}
            </section>
          ))}
          {ungroupedExperts.length > 0 ? (
            <section className="space-y-3">
              <div className="flex items-baseline gap-2">
                <Text variant="h4">No pod</Text>
                <Text variant="small" className="text-zinc-500">
                  {ungroupedExperts.length}{" "}
                  {ungroupedExperts.length === 1 ? "expert" : "experts"}
                </Text>
              </div>
              <div className={GRID_CLASS}>
                {ungroupedExperts.map(renderCard)}
              </div>
            </section>
          ) : null}
        </div>
      )}

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
