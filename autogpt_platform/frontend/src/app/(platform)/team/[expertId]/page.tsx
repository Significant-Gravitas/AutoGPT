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
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import {
  ArrowLeft02Icon,
  Briefcase01Icon,
  Calendar03Icon,
  CheckListIcon,
  PlugSocketIcon,
  Settings01Icon,
  SparklesIcon,
  UserIcon,
  WorkflowSquare01Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { notFound, useParams, useRouter } from "next/navigation";
import { FireExpertDialog } from "../components/FireExpertDialog/FireExpertDialog";
import { SoulDrawer } from "../components/SoulDrawer/SoulDrawer";
import { getLastRunLabel } from "../helpers";
import { ExpertAboutSection } from "./components/ExpertAboutSection";
import { ExpertDetailHeader } from "./components/ExpertDetailHeader";
import { ExpertIntegrationsSection } from "./components/ExpertIntegrationsSection/ExpertIntegrationsSection";
import { ExpertSchedulesSection } from "./components/ExpertSchedulesSection";
import { ExpertSettingsSection } from "./components/ExpertSettingsSection";
import { ExpertSkillsSection } from "./components/ExpertSkillsSection";
import { ExpertTasksSection } from "./components/ExpertTasksSection/ExpertTasksSection";
import { ExpertWorkSection } from "./components/ExpertWorkSection/ExpertWorkSection";
import { ExpertWorkflowsSection } from "./components/ExpertWorkflowsSection";
import { useExpertDetailPage } from "./useExpertDetailPage";

const MAIN_CLASS =
  "container min-h-screen space-y-6 pb-20 pt-8 sm:px-8 md:px-12";

const TABS = [
  { value: "basics", label: "Basics", icon: UserIcon },
  { value: "tasks", label: "Tasks", icon: CheckListIcon },
  { value: "work", label: "Work", icon: Briefcase01Icon },
  { value: "schedules", label: "Schedules", icon: Calendar03Icon },
  { value: "workflows", label: "Workflows", icon: WorkflowSquare01Icon },
  { value: "integrations", label: "Integrations", icon: PlugSocketIcon },
  { value: "skills", label: "Skills", icon: SparklesIcon },
  { value: "settings", label: "Settings", icon: Settings01Icon },
] as const;

function BackToTeamLink() {
  return (
    <Link
      href="/team"
      className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
      data-testid="expert-back-to-team"
    >
      <Icon icon={ArrowLeft02Icon} size={14} />
      Back to Team
    </Link>
  );
}

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
    openSoul,
    closeSoul,
  } = useExpertDetailPage({
    expertId,
    enabled: Boolean(enabled) && ready,
  });

  if (!ready || isLoading) {
    return (
      <main className={MAIN_CLASS}>
        <Skeleton className="h-20 w-full rounded-2xl" />
        <Skeleton className="h-10 w-full rounded-2xl" />
        <Skeleton className="h-48 w-full rounded-2xl" />
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

  const accent = getRaisedExpertAccent(expert.role, expert.color);
  const isPaused = Boolean(expert.schedules_paused_at);

  return (
    <main className={MAIN_CLASS}>
      <BackToTeamLink />
      <ExpertDetailHeader expert={expert} />

      {isPaused ? (
        <div className="flex items-center justify-between gap-2 rounded-xl bg-amber-50 px-4 py-3 ring-1 ring-inset ring-amber-200">
          <Text variant="small" className="text-amber-700">
            Schedules paused
          </Text>
          <Button
            variant="secondary"
            size="small"
            loading={isResuming}
            onClick={resumeSchedules}
          >
            Resume schedules
          </Button>
        </div>
      ) : null}

      <TabsLine defaultValue="basics">
        <TabsLineList flush className="overflow-x-auto">
          {TABS.map((tab) => (
            <TabsLineTrigger
              key={tab.value}
              value={tab.value}
              className="gap-2"
            >
              <Icon icon={tab.icon} size={16} />
              {tab.label}
            </TabsLineTrigger>
          ))}
        </TabsLineList>

        <TabsLineContent value="basics">
          <ExpertAboutSection text={expert.bio || expert.identity} />
        </TabsLineContent>

        <TabsLineContent value="tasks">
          <ExpertTasksSection
            expertId={expert.id}
            enabled={Boolean(enabled) && ready}
          />
        </TabsLineContent>

        <TabsLineContent value="work">
          <ExpertWorkSection
            expertId={expert.id}
            enabled={Boolean(enabled) && ready}
          />
        </TabsLineContent>

        <TabsLineContent value="schedules">
          <ExpertSchedulesSection
            expertName={expert.name}
            schedules={schedules}
            lastRunLabel={getLastRunLabel(expert)}
          />
        </TabsLineContent>

        <TabsLineContent value="workflows">
          <ExpertWorkflowsSection
            expert={expert}
            accentIconClass={accent.icon}
            onInstallWorkflow={openPicker}
          />
        </TabsLineContent>

        <TabsLineContent value="integrations">
          <ExpertIntegrationsSection
            expertId={expert.id}
            expertName={expert.name}
            expertAvatarUrl={expert.avatar_url ?? null}
          />
        </TabsLineContent>

        <TabsLineContent value="skills">
          <ExpertSkillsSection
            expertName={expert.name}
            skills={expert.skills}
          />
        </TabsLineContent>

        <TabsLineContent value="settings">
          <ExpertSettingsSection
            expert={expert}
            onEditSoul={openSoul}
            onFire={openFire}
          />
        </TabsLineContent>
      </TabsLine>

      <InstallWorkflowPicker
        mode="pick-workflow"
        expertId={expert.id}
        open={isPickerOpen}
        onClose={closePicker}
      />

      <SoulDrawer
        key={soulDrawerKey}
        expert={isSoulOpen ? expert : null}
        onClose={closeSoul}
      />

      <FireExpertDialog
        expertId={expert.id}
        expertName={expert.name}
        open={isFireOpen}
        onClose={closeFire}
        onFired={() => router.push("/team")}
      />
    </main>
  );
}
