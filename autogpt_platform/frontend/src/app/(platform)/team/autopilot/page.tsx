"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import {
  Calendar03Icon,
  SparklesIcon,
  UserIcon,
  WorkflowSquare01Icon,
} from "@hugeicons/core-free-icons";
import { notFound } from "next/navigation";
import { ExpertSchedulesSection } from "../[expertId]/components/ExpertSchedulesSection";
import { TeamSkillsSection } from "./components/TeamSkillsSection";
import { BackToTeamLink } from "../components/BackToTeamLink";
import { AutopilotAboutSection } from "./components/AutopilotAboutSection";
import { AutopilotHeader } from "./components/AutopilotHeader";
import { AutopilotSummaryCard } from "./components/AutopilotSummaryCard";
import { AutopilotWorkflowsSection } from "./components/AutopilotWorkflowsSection";
import { useAutopilotPage } from "./useAutopilotPage";

const MAIN_CLASS =
  "container min-h-screen max-w-[1180px] space-y-5 pb-16 pt-6 sm:px-8 md:px-12";

const TABS = [
  { value: "basics", label: "Basics", icon: UserIcon },
  { value: "schedules", label: "Schedules", icon: Calendar03Icon },
  { value: "workflows", label: "Workflows", icon: WorkflowSquare01Icon },
  { value: "skills", label: "Skills", icon: SparklesIcon },
] as const;

export default function AutopilotPage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const { experts, schedules, skills, isLoading, isError, refetch } =
    useAutopilotPage({ enabled: Boolean(enabled) && ready });

  if (!ready || isLoading) {
    return (
      <main className={MAIN_CLASS}>
        <Skeleton className="h-20 w-full rounded-xl" />
        <Skeleton className="h-10 w-full rounded-xl" />
        <Skeleton className="h-48 w-full rounded-xl" />
      </main>
    );
  }

  if (!enabled) {
    notFound();
  }

  if (isError) {
    return (
      <main className={MAIN_CLASS}>
        <BackToTeamLink />
        <ErrorCard
          context="Autopilot"
          hint="We could not load your team."
          onRetry={() => refetch()}
        />
      </main>
    );
  }

  const workflowCount = experts.reduce(
    (total, expert) => total + expert.workflows.length,
    0,
  );

  return (
    <main className={MAIN_CLASS}>
      <BackToTeamLink />
      <AutopilotHeader />

      <TabsLine defaultValue="basics">
        <TabsLineList
          flush
          className="overflow-x-auto"
          indicatorClassName="bg-zinc-900"
        >
          {TABS.map((tab) => (
            <TabsLineTrigger
              key={tab.value}
              value={tab.value}
              className="gap-1.5 px-2.5 py-2 text-[13px] leading-5 data-[state=active]:text-zinc-900"
            >
              <Icon icon={tab.icon} size={14} />
              {tab.label}
            </TabsLineTrigger>
          ))}
        </TabsLineList>

        <TabsLineContent value="basics">
          <div className="grid gap-4 lg:grid-cols-[3fr_1fr]">
            <AutopilotAboutSection />
            <AutopilotSummaryCard
              experts={experts}
              scheduleCount={schedules.length}
              skillCount={skills.length}
              workflowCount={workflowCount}
            />
          </div>
        </TabsLineContent>

        <TabsLineContent value="schedules">
          <ExpertSchedulesSection
            title="Team schedules"
            expertName="your team"
            schedules={schedules}
            lastRunLabel={null}
          />
        </TabsLineContent>

        <TabsLineContent value="workflows">
          <AutopilotWorkflowsSection experts={experts} />
        </TabsLineContent>

        <TabsLineContent value="skills">
          <TeamSkillsSection expertName="Autopilot" skills={skills} />
        </TabsLineContent>
      </TabsLine>
    </main>
  );
}
