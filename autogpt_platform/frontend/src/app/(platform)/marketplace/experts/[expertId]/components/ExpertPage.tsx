"use client";

import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { ReactNode } from "react";
import { ExpertAbout } from "./ExpertAbout";
import { ExpertComingSoonLabel } from "./ExpertComingSoonLabel";
import { ExpertHireActions } from "./ExpertHireActions";
import { ExpertPageHeader } from "./ExpertPageHeader";
import { ExpertSkills } from "./ExpertSkills";
import { ExpertWorkflowList } from "./ExpertWorkflowList";
import { useExpertPage } from "../useExpertPage";
import { useHireFlow } from "../useHireFlow";

const MAIN_CLASS =
  "mx-auto flex w-full max-w-[760px] flex-col px-6 pb-24 pt-8 md:px-8";

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
  const {
    expert,
    hiredExpert,
    isLoggedIn,
    isHiringOpen,
    isActionReady,
    isLoading,
    isError,
    refetch,
  } = useExpertPage({ expertId });
  const {
    hire,
    isHiring,
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

  let actions: ReactNode = <Skeleton className="h-9 w-28 rounded-full" />;
  if (isActionReady) {
    actions = isHiringOpen ? (
      <ExpertHireActions
        expert={expert}
        hiredExpert={hiredExpert}
        isLoggedIn={isLoggedIn}
        isHiring={isHiring}
        onHire={hire}
      />
    ) : (
      <ExpertComingSoonLabel />
    );
  }

  return (
    <main className={MAIN_CLASS}>
      <BackToMarketplaceLink />
      <ExpertPageHeader expert={expert} accent={accent} actions={actions} />
      <div className="mt-8 flex flex-col gap-10 border-t border-zinc-200 pt-8">
        <ExpertAbout key={expert.id} text={expert.bio || expert.identity} />
        <ExpertSkills skills={expert.skills ?? []} accent={accent} />
        <ExpertWorkflowList
          name={expert.name}
          workflows={expert.workflows}
          accent={accent}
        />
      </div>

      {/* The voice pick follows a successful hire when the persona ships
          writing samples; dismissing it still celebrates the hire. */}
      <Dialog
        styling={{ width: "640px" }}
        controlled={{
          isOpen: hireResult !== null,
          set: (open) => {
            if (!open) dismissVoicePick();
          },
        }}
      >
        <Dialog.Content>
          {hireResult ? (
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
