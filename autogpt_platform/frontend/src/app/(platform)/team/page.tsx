"use client";

import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import { cn } from "@/lib/utils";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { notFound } from "next/navigation";
import { AutopilotCard } from "./components/AutopilotCard";
import { CreateMenu } from "./components/CreateMenu/CreateMenu";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertTeamCard } from "./components/ExpertTeamCard/ExpertTeamCard";
import { ExpertTeamCardSkeleton } from "./components/ExpertTeamCardSkeleton";
import { SoulDrawer } from "./components/SoulDrawer/SoulDrawer";
import { WhatRunsZone } from "./components/WhatRunsZone/WhatRunsZone";
import { SECTION_INSET_CLASS } from "./helpers";
import { useTeamPage } from "./useTeamPage";

const MAIN_CLASS =
  "container min-h-screen space-y-6 pb-20 pt-8 sm:px-8 md:px-12";
// Auto-fill so a wider row simply takes another card instead of stretching them.
const GRID_CLASS =
  "grid grid-cols-[repeat(auto-fill,minmax(19rem,1fr))] gap-6 [&>*]:max-w-[24rem]";

export default function TeamPage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const {
    hiredExperts,
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
  } = useTeamPage({ enabled: Boolean(enabled) && ready });

  if (!ready) {
    return (
      <main className={MAIN_CLASS}>
        <div className={GRID_CLASS}>
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
        <CreateMenu />
      </div>
      <div className={GRID_CLASS}>
        <AutopilotCard />
        {isLoading
          ? [0, 1, 2].map((i) => <ExpertTeamCardSkeleton key={i} />)
          : hiredExperts.map((expert) => (
              <ExpertTeamCard
                key={expert.id}
                expert={expert}
                schedules={schedulesForExpert(expert)}
                onInstallWorkflow={installWorkflow}
                onEditSoul={openSoul}
              />
            ))}
      </div>
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
      {!isLoading && !isError && hiredExperts.length === 0 ? (
        <EmptyTeamState />
      ) : null}
      <InstallWorkflowPicker
        mode="pick-workflow"
        expertId={pickerExpertId ?? undefined}
        open={pickerExpertId !== null}
        onClose={closeWorkflowPicker}
      />
      <SoulDrawer key={soulDrawerKey} expert={soulExpert} onClose={closeSoul} />
    </main>
  );
}
