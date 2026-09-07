"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
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
import { ExpertWorkflowsSection } from "../[expertId]/components/ExpertWorkflowsSection";
import { BackToTeamLink } from "../components/BackToTeamLink";
import { AUTOPILOT_PILL_CLASS } from "../helpers";
import { AutopilotAboutSection } from "./components/AutopilotAboutSection";
import { AutopilotHeader } from "./components/AutopilotHeader";
import { AutopilotSkillsSection } from "./components/AutopilotSkillsSection";
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
  const { schedules, workflows, skills, isLoading, isError, refetch } =
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

  return (
    <main className={MAIN_CLASS}>
      <BackToTeamLink />
      <AutopilotHeader />

      <Text variant="body" tone="muted">
        Built in, always on your team.
      </Text>

      <TabsLine variant="compact" defaultValue="basics">
        <TabsLineList className="overflow-x-auto">
          {TABS.map((tab) => (
            <TabsLineTrigger key={tab.value} value={tab.value} icon={tab.icon}>
              {tab.label}
            </TabsLineTrigger>
          ))}
        </TabsLineList>

        <TabsLineContent value="basics">
          <AutopilotAboutSection />
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
          <ExpertWorkflowsSection
            expertName="Autopilot"
            workflows={workflows}
            accentClassName={AUTOPILOT_PILL_CLASS}
            emptyMessage="No workflows yet. Workflows in your library that no expert owns show up here."
          />
        </TabsLineContent>

        <TabsLineContent value="skills">
          <AutopilotSkillsSection skills={skills} />
        </TabsLineContent>
      </TabsLine>
    </main>
  );
}
