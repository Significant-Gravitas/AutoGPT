"use client";

import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { ExpertAbout } from "./components/ExpertAbout";
import { ExpertComingSoon } from "./components/ExpertComingSoon";
import { ExpertHireCard } from "./components/ExpertHireCard";
import { ExpertPageHeader } from "./components/ExpertPageHeader";
import { ExpertSkills } from "./components/ExpertSkills";
import { ExpertWorkflowList } from "./components/ExpertWorkflowList";
import { useExpertPage } from "./useExpertPage";
import { useHireFlow } from "./useHireFlow";

const MAIN_CLASS =
  "mx-auto flex w-full max-w-[1120px] flex-col gap-8 px-6 pb-20 pt-8 md:px-10";

function BackToMarketplaceLink() {
  return (
    <Link
      href="/marketplace#experts"
      className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
    >
      <Icon icon={ArrowLeft02Icon} size={14} />
      Back to marketplace
    </Link>
  );
}

export default function MarketplaceExpertPage() {
  const { expertId } = useParams<{ expertId: string }>();
  const { isLoggedIn } = useAuth();
  const { expert, hiredExpert, canHire, isReady, isLoading, isError, refetch } =
    useExpertPage({ expertId });
  const {
    hire,
    isHiring,
    hireResult,
    pickVoice,
    skipVoice,
    dismissVoicePick,
    isSavingVoice,
  } = useHireFlow(expert);

  if (!isReady || isLoading) {
    return (
      <main className={MAIN_CLASS}>
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-44 w-full rounded-3xl" />
        <div className="grid gap-8 lg:grid-cols-[1fr_320px]">
          <Skeleton className="h-72 w-full rounded-2xl" />
          <Skeleton className="h-56 w-full rounded-3xl" />
        </div>
      </main>
    );
  }

  if (!canHire) {
    return (
      <main className={MAIN_CLASS}>
        <BackToMarketplaceLink />
        <ExpertComingSoon isLoggedIn={isLoggedIn} />
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

  return (
    <main className={MAIN_CLASS}>
      <BackToMarketplaceLink />
      <ExpertPageHeader expert={expert} accent={accent} />
      <div className="grid gap-8 lg:grid-cols-[1fr_320px] lg:items-start">
        <div className="flex min-w-0 flex-col gap-10">
          <ExpertAbout key={expert.id} text={expert.bio || expert.identity} />
          <ExpertSkills skills={expert.skills ?? []} />
          <ExpertWorkflowList workflows={expert.workflows} accent={accent} />
        </div>
        <ExpertHireCard
          expert={expert}
          accent={accent}
          hiredExpert={hiredExpert}
          isHiring={isHiring}
          onHire={hire}
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
