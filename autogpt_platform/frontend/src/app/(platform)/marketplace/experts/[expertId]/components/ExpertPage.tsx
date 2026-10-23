"use client";

import {
  getExpertAccent,
  getExpertAccessProviders,
  getExpertFirstName,
} from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import { useTrackExpertProfileOpened } from "@/services/experts/use-track-funnel-view-once";
import { useHireFlow } from "@/services/experts/useHireFlow";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { ReactNode } from "react";
import { useExpertPage } from "../useExpertPage";
import { ExpertAbout } from "./ExpertAbout";
import { ExpertAccess } from "./ExpertAccess";
import { ExpertDayOne } from "./ExpertDayOne";
import { ExpertHireActions } from "./ExpertHireActions";
import { ExpertHireCard } from "./ExpertHireCard";
import { ExpertHeroBand, ExpertHeroIdentity } from "./ExpertPageHeader";
import { ExpertProtectedRules } from "./ExpertProtectedRules";
import { ExpertSkills } from "./ExpertSkills";
import { ExpertWorkflowList } from "./ExpertWorkflowList";

const MAIN_CLASS =
  "mx-auto flex w-full max-w-[1120px] flex-col px-4 pb-24 pt-8 sm:px-6 md:px-8";

function BackToMarketplaceLink() {
  return (
    <Link
      href="/marketplace#experts"
      className="mb-6 inline-flex w-fit items-center gap-1.5 text-[13px] text-zinc-500 transition-colors hover:text-zinc-900"
    >
      <Icon icon={ArrowLeft02Icon} size={14} />
      Back to marketplace
    </Link>
  );
}

export function ExpertPage() {
  const { expertId } = useParams<{ expertId: string }>();
  useTrackExpertProfileOpened(expertId);
  const {
    expert,
    hiredExpert,
    isLoggedIn,
    isActionReady,
    systemProviders,
    isLoading,
    isError,
    refetch,
  } = useExpertPage({ expertId });
  const {
    hire,
    isHiring,
    isVoicePickOpen,
    hireResult,
    pickVoice,
    skipVoice,
    dismissVoicePick,
    isSavingVoice,
  } = useHireFlow(expert);

  if (isLoading) {
    return (
      <main className={MAIN_CLASS}>
        <Skeleton className="mb-6 h-4 w-32" />
        <div className="flex items-center gap-5">
          <Skeleton className="h-18 w-18 rounded-full" />
          <div className="flex flex-1 flex-col gap-2.5">
            <Skeleton className="h-7 w-36" />
            <Skeleton className="h-5 w-24 rounded-md" />
          </div>
          <Skeleton className="h-9 w-28 rounded-full" />
        </div>
        <Skeleton className="mt-5 h-5 w-3/4" />
        <div className="mt-8 flex flex-col gap-3 border-t border-zinc-200 pt-8">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-11/12" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      </main>
    );
  }

  if (isError) {
    return (
      <main className={MAIN_CLASS}>
        <BackToMarketplaceLink />
        <ErrorCard
          context="this expert"
          hint="We could not load this expert."
          onRetry={() => refetch()}
        />
      </main>
    );
  }

  if (!expert) {
    notFound();
  }

  const accent = getExpertAccent(expert.role);
  const firstName = getExpertFirstName(expert.name);
  const accessProviders = getExpertAccessProviders(
    expert.workflows,
    systemProviders,
  );
  const stats = getExpertStats(expert, accessProviders.length);

  let actions: ReactNode = <Skeleton className="h-9 w-full rounded-full" />;
  if (isActionReady) {
    actions = (
      <ExpertHireActions
        expert={expert}
        hiredExpert={hiredExpert}
        isLoggedIn={isLoggedIn}
        isHiring={isHiring}
        onHire={hire}
      />
    );
  }

  return (
    <main className={MAIN_CLASS}>
      <BackToMarketplaceLink />
      <ExpertHeroBand expert={expert} />
      {/* One grid for the hero's text, the hire card and the story. On wide
          screens the card spans the right column, pulled up over the band's
          end and sticky while the story scrolls; on phones it sits between
          the hero text and the story. */}
      <div className="grid gap-x-10 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="order-1 min-w-0 lg:col-start-1 lg:row-start-1">
          <ExpertHeroIdentity expert={expert} stats={stats} />
        </div>
        <div className="relative z-10 order-2 mt-8 lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:-mt-24">
          <div className="lg:sticky lg:top-6">
            <ExpertHireCard
              expert={expert}
              firstName={firstName}
              actions={actions}
            />
          </div>
        </div>
        <div className="order-3 mt-12 flex min-w-0 flex-col gap-12 lg:col-start-1 lg:row-start-2">
          <ExpertDayOne
            name={firstName}
            items={expert.day_one ?? []}
            accent={accent}
          />
          <ExpertAbout key={expert.id} text={expert.bio || expert.identity} />
          <ExpertSkills
            skills={expert.bundled_skills ?? []}
            category={expert.categories?.[0]}
          />
          <ExpertAccess name={firstName} providers={accessProviders} />
          <ExpertProtectedRules rules={expert.protected_soul_rules ?? []} />
          <ExpertWorkflowList name={expert.name} workflows={expert.workflows} />
        </div>
      </div>

      {/* The voice pick follows a successful hire when the persona ships
          writing samples; dismissing it still celebrates the hire. */}
      <Dialog
        styling={{ width: "640px" }}
        controlled={{
          isOpen: isVoicePickOpen,
          set: (open) => {
            if (!open) dismissVoicePick();
          },
        }}
      >
        <Dialog.Content>
          {isVoicePickOpen && hireResult ? (
            <VoicePicker
              name={hireResult.expert.name}
              samples={expert.voice_samples ?? []}
              onPick={pickVoice}
              onSkip={skipVoice}
              isSubmitting={isSavingVoice}
            />
          ) : null}
        </Dialog.Content>
      </Dialog>
    </main>
  );
}

function plural(count: number, word: string) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** The hero's at-a-glance chips: only the counts that are not zero. */
function getExpertStats(
  expert: { bundled_skills?: unknown[] | null; workflows: unknown[] },
  serviceCount: number,
) {
  const skillCount = expert.bundled_skills?.length ?? 0;
  return [
    skillCount > 0 ? plural(skillCount, "skill") : null,
    expert.workflows.length > 0
      ? plural(expert.workflows.length, "workflow")
      : null,
    serviceCount > 0 ? plural(serviceCount, "service") : null,
  ].filter((stat): stat is string => stat !== null);
}
