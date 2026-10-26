"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { AUTOPILOT_HEX } from "@/components/molecules/ExpertAvatar/colors";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import {
  Calendar03Icon,
  Settings01Icon,
  SparklesIcon,
  TaskDone01Icon,
  UserIcon,
  WorkflowSquare01Icon,
} from "@hugeicons/core-free-icons";
import { ExpertSchedulesSection } from "../[expertId]/components/ExpertSchedulesSection";
import { ExpertWorkflowsSection } from "../[expertId]/components/ExpertWorkflowsSection";
import { BackToTeamLink } from "../components/BackToTeamLink";
import {
  PROFILE_TABS_INDICATOR_CLASS,
  PROFILE_TABS_LIST_CLASS,
  PROFILE_TABS_TRIGGER_CLASS,
} from "../helpers";
import { AutopilotAboutSection } from "./components/AutopilotAboutSection";
import { AutopilotDelegationsSection } from "./components/AutopilotDelegationsSection/AutopilotDelegationsSection";
import { DelegationSettingsSection } from "./components/DelegationSettingsSection/DelegationSettingsSection";
import { AutopilotHeader } from "./components/AutopilotHeader";
import { AutopilotSkillsSection } from "./components/AutopilotSkillsSection";
import { useAutopilotDelegations } from "./useAutopilotDelegations";
import { useAutopilotPage } from "./useAutopilotPage";
import { useAutopilotTab } from "./useAutopilotTab";

const MAIN_CLASS =
  "container min-h-screen max-w-[1180px] space-y-5 px-4 pb-16 pt-6 sm:px-8 md:px-12";

const TABS = [
  { value: "basics", label: "Basics", icon: UserIcon },
  { value: "delegations", label: "Delegations", icon: TaskDone01Icon },
  { value: "schedules", label: "Schedules", icon: Calendar03Icon },
  { value: "workflows", label: "Workflows", icon: WorkflowSquare01Icon },
  { value: "skills", label: "Skills", icon: SparklesIcon },
  { value: "settings", label: "Settings", icon: Settings01Icon },
] as const;

export default function AutopilotPage() {
  const { activeTab, onTabChange } = useAutopilotTab();
  const { schedules, workflows, skills, isLoading, isError, refetch } =
    useAutopilotPage();
  const delegations = useAutopilotDelegations();

  if (isLoading) {
    return (
      <main className={MAIN_CLASS}>
        <Skeleton className="h-20 w-full rounded-xl" />
        <Skeleton className="h-10 w-full rounded-xl" />
        <Skeleton className="h-48 w-full rounded-xl" />
      </main>
    );
  }

  if (isError) {
    return (
      <main className={MAIN_CLASS}>
        <BackToTeamLink />
        <ErrorCard
          context={AUTOPILOT_NAME}
          hint="We could not load your team."
          onRetry={() => refetch()}
        />
      </main>
    );
  }

  return (
    <main className={MAIN_CLASS}>
      <BackToTeamLink />
      <AutopilotHeader delegationsToday={delegations.todayCount} />

      <Text variant="body" tone="muted">
        Built in, always on your team.
      </Text>

      <TabsLine variant="compact" value={activeTab} onValueChange={onTabChange}>
        <TabsLineList
          flush={false}
          className={PROFILE_TABS_LIST_CLASS}
          indicatorClassName={PROFILE_TABS_INDICATOR_CLASS}
        >
          {TABS.map((tab) => (
            <TabsLineTrigger
              className={PROFILE_TABS_TRIGGER_CLASS}
              key={tab.value}
              value={tab.value}
              icon={tab.icon}
            >
              <span className="max-sm:sr-only">{tab.label}</span>
            </TabsLineTrigger>
          ))}
        </TabsLineList>

        <TabsLineContent value="basics">
          <AutopilotAboutSection />
        </TabsLineContent>

        <TabsLineContent value="delegations">
          <AutopilotDelegationsSection
            pages={delegations}
            todayCount={delegations.todayCount}
            mode={delegations.mode}
            onChangeMode={() => onTabChange("settings")}
          />
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
            expertName={AUTOPILOT_NAME}
            workflows={workflows}
            coverColor={AUTOPILOT_HEX}
            emptyMessage="Workflows in your library that no expert owns show up here."
          />
        </TabsLineContent>

        <TabsLineContent value="skills">
          <AutopilotSkillsSection skills={skills} />
        </TabsLineContent>

        <TabsLineContent value="settings">
          <DelegationSettingsSection />
        </TabsLineContent>
      </TabsLine>
    </main>
  );
}
