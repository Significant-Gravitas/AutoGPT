"use client";

import { ExpertProfileSheet } from "@/app/(platform)/marketplace/components/ExpertsSection/components/ExpertProfileSheet/ExpertProfileSheet";
import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { notFound } from "next/navigation";
import { AutopilotCard } from "./components/AutopilotCard";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertTeamCard } from "./components/ExpertTeamCard/ExpertTeamCard";
import { useTeamPage } from "./useTeamPage";

const MAIN_CLASS =
  "container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12";
const GRID_CLASS = "grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3";

export default function TeamPage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const {
    hiredExperts,
    isLoading,
    isError,
    refetch,
    installWorkflow,
    pickerExpertId,
    closeWorkflowPicker,
    profileExpert,
    openProfile,
    closeProfile,
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

  return (
    <main className={MAIN_CLASS}>
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2.5">
          <AITeamIcon size={36} className="shrink-0 text-black" />
          <Text variant="h3">Your Team</Text>
        </div>
        <Text variant="body" className="max-w-prose text-zinc-600">
          Autopilot and your hired experts, ready to work.
        </Text>
      </div>
      <div className={GRID_CLASS}>
        <AutopilotCard />
        {isLoading
          ? [0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-48 w-full rounded-2xl" />
            ))
          : hiredExperts.map((expert) => (
              <ExpertTeamCard
                key={expert.id}
                expert={expert}
                onInstallWorkflow={installWorkflow}
                onOpenProfile={openProfile}
              />
            ))}
      </div>
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
      <ExpertProfileSheet
        expert={profileExpert}
        onClose={closeProfile}
        presentation="drawer"
      />
    </main>
  );
}
