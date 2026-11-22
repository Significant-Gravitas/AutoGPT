"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import { cn } from "@/lib/utils";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertChatDrawer } from "./components/ExpertChatDrawer/ExpertChatDrawer";
import { ExpertTeamCard } from "./components/ExpertTeamCard/ExpertTeamCard";
import { SetupNeeded } from "./components/SetupNeeded/SetupNeeded";
import { SoulDrawer } from "./components/SoulDrawer/SoulDrawer";
import { TeamPageHeader } from "./components/TeamPageHeader/TeamPageHeader";
import { TeamRoster } from "./components/TeamRoster/TeamRoster";
import { TeamRosterToolbar } from "./components/TeamRoster/TeamRosterToolbar";
import { useTeamRosterView } from "./components/TeamRoster/useTeamRosterView";
import { useTeamPage } from "./useTeamPage";

const MAIN_CLASS =
  "mx-auto min-h-screen w-full max-w-[1180px] space-y-5 px-4 pb-16 pt-6 duration-500 sm:px-8 md:px-12 animate-in fade-in slide-in-from-bottom-2 fill-mode-both motion-reduce:animate-none";

export default function TeamPage() {
  const {
    hiredExperts,
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
  } = useTeamPage();
  const roster = useTeamRosterView({
    experts: hiredExperts,
    schedulesForExpert,
  });

  function renderCard(expert: Expert) {
    return (
      <ExpertTeamCard
        key={expert.id}
        expert={expert}
        schedules={schedulesForExpert(expert)}
        onInstallWorkflow={installWorkflow}
        onEditSoul={openSoul}
        onChat={openChat}
      />
    );
  }

  return (
    <div className="flex w-full">
      <main className={cn(MAIN_CLASS, "min-w-0 flex-1")}>
        <TeamPageHeader experts={hiredExperts} />

        {isError ? (
          <ErrorCard
            context="your team"
            hint="We could not load your hired experts."
            onRetry={() => refetch()}
          />
        ) : null}

        <TeamRosterToolbar
          query={roster.query}
          onQueryChange={roster.setQuery}
          filter={roster.filter}
          onFilterChange={roster.setFilter}
        />

        <div className="space-y-6">
          <SetupNeeded />
          <TeamRoster
            isLoading={isLoading}
            experts={hiredExperts}
            visibleExperts={roster.visibleExperts}
            isNarrowed={roster.isNarrowed}
            schedulesForExpert={schedulesForExpert}
            renderCard={renderCard}
            onAutopilotChat={() => openChat(null)}
          />

          {!isLoading && !isError && hiredExperts.length === 0 ? (
            <EmptyTeamState />
          ) : null}
        </div>

        <InstallWorkflowPicker
          mode="pick-workflow"
          expertId={pickerExpertId ?? undefined}
          open={pickerExpertId !== null}
          onClose={closeWorkflowPicker}
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
