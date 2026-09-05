"use client";

import { getRaisedExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
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
import {
  Briefcase01Icon,
  Calendar03Icon,
  PlugSocketIcon,
  Settings01Icon,
  SparklesIcon,
  UserIcon,
  WorkflowSquare01Icon,
} from "@hugeicons/core-free-icons";
import { notFound, useParams, useRouter } from "next/navigation";
import { BackToTeamLink } from "../components/BackToTeamLink";
import { FireExpertDialog } from "../components/FireExpertDialog/FireExpertDialog";
import { SoulDrawer } from "../components/SoulDrawer/SoulDrawer";
import { getLastRunLabel } from "../helpers";
import { ExpertAboutSection } from "./components/ExpertAboutSection";
import { ExpertBudgetSection } from "./components/ExpertBudgetSection";
import { ExpertDetailHeader } from "./components/ExpertDetailHeader";
import { ExpertIntegrationsSection } from "./components/ExpertIntegrationsSection/ExpertIntegrationsSection";
import { ExpertSchedulesSection } from "./components/ExpertSchedulesSection";
import { ExpertSettingsSection } from "./components/ExpertSettingsSection";
import { ExpertSkillsSection } from "./components/ExpertSkillsSection";
import { ExpertSummaryCard } from "./components/ExpertSummaryCard";
import { ExpertWorkSection } from "./components/ExpertWorkSection/ExpertWorkSection";
import { ExpertWorkflowsSection } from "./components/ExpertWorkflowsSection";
import { useExpertDetailPage } from "./useExpertDetailPage";
import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";

const MAIN_CLASS =
  "container min-h-screen max-w-[1180px] space-y-5 pb-16 pt-6 sm:px-8 md:px-12";

const TABS = [
  { value: "basics", label: "Basics", icon: UserIcon },
  { value: "work", label: "Work", icon: Briefcase01Icon },
  { value: "schedules", label: "Schedules", icon: Calendar03Icon },
  { value: "workflows", label: "Workflows", icon: WorkflowSquare01Icon },
  { value: "integrations", label: "Integrations", icon: PlugSocketIcon },
  { value: "skills", label: "Skills", icon: SparklesIcon },
  { value: "settings", label: "Settings", icon: Settings01Icon },
] as const;

export default function ExpertDetailPage() {
  const { expertId } = useParams<{ expertId: string }>();
  const router = useRouter();
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const {
    expert,
    isLoading,
    isError,
    refetch,
    schedules,
    activity,
    isActivityLoading,
    isActivityError,
    isPickerOpen,
    openPicker,
    closePicker,
    resumeSchedules,
    isResuming,
    isFireOpen,
    openFire,
    closeFire,
    isSoulOpen,
    soulDrawerKey,
    toggleSoul,
    closeSoul,
  } = useExpertDetailPage({
    expertId,
    enabled: Boolean(enabled) && ready,
  });

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

  if (isError || !expert) {
    return (
      <main className={MAIN_CLASS}>
        <BackToTeamLink />
        <ErrorCard
          context="this expert"
          hint="We could not load this expert."
          onRetry={() => refetch()}
        />
      </main>
    );
  }

  const isPaused = Boolean(expert.schedules_paused_at);

  return (
    <div className="flex w-full">
      <main className={cn(MAIN_CLASS, "min-w-0 flex-1")}>
        <BackToTeamLink />
        <ExpertDetailHeader expert={expert} onEditSoul={toggleSoul} />

        {isPaused ? (
          <div className="flex items-center justify-between gap-2 rounded-lg bg-amber-50 px-4 py-2.5 ring-1 ring-inset ring-amber-200">
            <Text variant="small" className="text-amber-700">
              Schedules paused
            </Text>
            <Button
              variant="secondary"
              size="small"
              className={ACTION_BUTTON_CLASS}
              loading={isResuming}
              onClick={resumeSchedules}
            >
              Resume schedules
            </Button>
          </div>
        ) : null}

        <ExpertBudgetSection expert={expert} />

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
                className="gap-1.5 px-2.5 py-2 text-xs leading-5 data-[state=active]:text-zinc-900"
              >
                <Icon icon={tab.icon} size={14} />
                {tab.label}
              </TabsLineTrigger>
            ))}
          </TabsLineList>

          <TabsLineContent value="basics">
            <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(300px,1fr)]">
              <ExpertAboutSection
                bio={expert.bio}
                identity={expert.identity}
                voicePreferences={expert.voice_preferences}
                boundaries={expert.boundaries}
              />
              <ExpertSummaryCard
                expert={expert}
                activity={activity}
                isActivityLoading={isActivityLoading}
                isActivityError={isActivityError}
              />
            </div>
          </TabsLineContent>

          <TabsLineContent value="work">
            <ExpertWorkSection
              expertId={expert.id}
              enabled={Boolean(enabled) && ready}
            />
          </TabsLineContent>

          <TabsLineContent value="schedules">
            <ExpertSchedulesSection
              title={`${expert.name}'s Schedules`}
              accentClassName={
                getRaisedExpertAccent(expert.role, expert.color).pill
              }
              expertName={expert.name}
              expertId={expert.id}
              workflows={expert.workflows}
              schedules={schedules}
              lastRunLabel={getLastRunLabel(expert)}
            />
          </TabsLineContent>

          <TabsLineContent value="workflows">
            <ExpertWorkflowsSection
              expert={expert}
              onInstallWorkflow={openPicker}
            />
          </TabsLineContent>

          <TabsLineContent value="integrations">
            <ExpertIntegrationsSection
              expertId={expert.id}
              expertName={expert.name}
            />
          </TabsLineContent>

          <TabsLineContent value="skills">
            <ExpertSkillsSection
              expert={expert}
              accentClassName={
                getRaisedExpertAccent(expert.role, expert.color).pill
              }
            />
          </TabsLineContent>

          <TabsLineContent value="settings">
            <ExpertSettingsSection expert={expert} onFire={openFire} />
          </TabsLineContent>
        </TabsLine>

        <InstallWorkflowPicker
          mode="pick-workflow"
          expertId={expert.id}
          open={isPickerOpen}
          onClose={closePicker}
        />

        <FireExpertDialog
          expertId={expert.id}
          expertName={expert.name}
          open={isFireOpen}
          onClose={closeFire}
          onFired={() => router.push("/team")}
        />
      </main>

      <SoulDrawer
        key={soulDrawerKey}
        expert={isSoulOpen ? expert : null}
        onClose={closeSoul}
      />
    </div>
  );
}
