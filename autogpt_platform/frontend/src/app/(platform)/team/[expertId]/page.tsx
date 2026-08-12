"use client";

import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InstallWorkflowPicker } from "@/components/molecules/InstallWorkflowPicker/InstallWorkflowPicker";
import { cn } from "@/lib/utils";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { getLastRunLabel } from "../helpers";
import { ExpertAboutSection } from "./components/ExpertAboutSection";
import { ExpertSchedulesSection } from "./components/ExpertSchedulesSection";
import { ExpertWorkflowsSection } from "./components/ExpertWorkflowsSection";
import { useExpertDetailPage } from "./useExpertDetailPage";

const MAIN_CLASS =
  "container min-h-screen space-y-8 pb-20 pt-16 sm:px-8 md:px-12";

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
  } = useExpertDetailPage({
    expertId,
    enabled: Boolean(enabled) && ready,
  });

  if (!ready || isLoading) {
    return (
      <main className={MAIN_CLASS}>
        <Skeleton className="h-32 w-full rounded-2xl" />
        <Skeleton className="h-48 w-full rounded-2xl" />
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

  const accent = getExpertAccent(expert.role);
  const isPaused = Boolean(expert.schedules_paused_at);

  return (
    <main className={MAIN_CLASS}>
      <BackToTeamLink />

      <header
        className={cn(
          "relative flex flex-col gap-5 overflow-hidden rounded-2xl border border-zinc-200/60 p-6 sm:flex-row sm:items-center",
          accent.washWide,
        )}
      >
        <Avatar className="h-20 w-20 bg-white shadow-sm ring-1 ring-black/5">
          {expert.avatar_url ? (
            <AvatarImage src={expert.avatar_url} alt={expert.name} />
          ) : null}
          <AvatarFallback>{expert.name}</AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-3xl font-semibold tracking-[-0.02em] text-zinc-900">
              {expert.name}
            </h1>
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium",
                accent.pill,
              )}
            >
              <Icon icon={accent.roleIcon} size={14} />
              {expert.role}
            </span>
          </div>
          {expert.tagline ? (
            <p className="mt-1.5 text-base text-zinc-500">{expert.tagline}</p>
          ) : null}
        </div>
        <div className="shrink-0">
          <Button
            as="NextLink"
            href={`/copilot?expertId=${expert.id}`}
            variant="primary"
            size="small"
          >
            Chat
          </Button>
        </div>
      </header>

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

      <ExpertAboutSection text={expert.bio || expert.identity} />

      {expert.skills && expert.skills.length > 0 ? (
        <section>
          <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
            Skills
          </div>
          <div className="flex flex-wrap gap-2">
            {expert.skills.map((skill) => (
              <span
                key={skill}
                className="rounded-full bg-zinc-50 px-3 py-1.5 text-sm text-zinc-600 ring-1 ring-inset ring-zinc-200/80"
              >
                {skill}
              </span>
            ))}
          </div>
        </section>
      ) : null}

      <ExpertSchedulesSection
        expertName={expert.name}
        schedules={schedules}
        lastRunLabel={getLastRunLabel(expert)}
      />

      <ExpertWorkflowsSection
        expert={expert}
        accentIconClass={accent.icon}
        onInstallWorkflow={openPicker}
      />

      <InstallWorkflowPicker
        mode="pick-workflow"
        expertId={expert.id}
        open={isPickerOpen}
        onClose={closePicker}
      />
    </main>
  );
}
