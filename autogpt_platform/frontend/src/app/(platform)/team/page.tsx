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
import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { notFound } from "next/navigation";
import { useState } from "react";
import { EmptyTeamState } from "./components/EmptyTeamState";
import { ExpertChatDrawer } from "./components/ExpertChatDrawer/ExpertChatDrawer";
import { ExpertTeamCard } from "./components/ExpertTeamCard/ExpertTeamCard";
import { ExpertTeamCardSkeleton } from "./components/ExpertTeamCardSkeleton";
import { SetupNeeded } from "./components/SetupNeeded/SetupNeeded";
import { SoulDrawer } from "./components/SoulDrawer/SoulDrawer";
import { TeamHeaderActions } from "./components/TeamHeaderActions";
import { TeamRoster } from "./components/TeamRoster/TeamRoster";
import { TeamRosterToolbar } from "./components/TeamRoster/TeamRosterToolbar";
import { useTeamRosterView } from "./components/TeamRoster/useTeamRosterView";
import { TEAM_GRID_CLASS } from "./helpers";
import { useTeamPage } from "./useTeamPage";

const MAIN_CLASS =
  "mx-auto min-h-screen w-full max-w-[1180px] space-y-5 px-4 pb-16 pt-6 duration-500 sm:px-8 md:px-12 animate-in fade-in slide-in-from-bottom-2 fill-mode-both motion-reduce:animate-none";

const TABS = [
  { value: "overview", label: "Overview", icon: UserGroupIcon },
] as const;

type TeamTab = (typeof TABS)[number]["value"];

export default function TeamPage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
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
  } = useTeamPage({ enabled: Boolean(enabled) && ready });
  const [tab, setTab] = useState<TeamTab>("overview");
  const roster = useTeamRosterView({
    experts: hiredExperts,
    schedulesForExpert,
  });

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
        onInstallWorkflow={installWorkflow}
        onEditSoul={openSoul}
        onChat={openChat}
      />
    );
  }

  return (
    <div className="flex w-full">
      <main className={cn(MAIN_CLASS, "min-w-0 flex-1")}>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex flex-col gap-1">
            <Text variant="lead-medium" as="h1" tone="primary">
              Team
            </Text>
            <Text variant="body" tone="secondary" className="max-w-prose">
              Autopilot and your hired experts, ready to work.
            </Text>
          </div>
          <TeamHeaderActions />
        </div>

        {isError ? (
          <ErrorCard
            context="your team"
            hint="We could not load your hired experts."
            onRetry={() => refetch()}
          />
        ) : null}

        <TabsLine
          variant="compact"
          value={tab}
          onValueChange={(next) => setTab(next as TeamTab)}
        >
          {/* The roster's search and filter share the tabs row, right-aligned,
              and only while the roster is the tab being shown. */}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <TabsLineList className="w-auto overflow-x-auto border-b-transparent">
              {TABS.map((item) => (
                <TabsLineTrigger
                  key={item.value}
                  value={item.value}
                  icon={item.icon}
                >
                  {item.label}
                </TabsLineTrigger>
              ))}
            </TabsLineList>
            {tab === "overview" ? (
              <TeamRosterToolbar
                query={roster.query}
                onQueryChange={roster.setQuery}
                filter={roster.filter}
                onFilterChange={roster.setFilter}
              />
            ) : null}
          </div>

          <TabsLineContent value="overview" className="space-y-6">
            <SetupNeeded enabled={Boolean(enabled) && ready} />
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
          </TabsLineContent>
        </TabsLine>

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
