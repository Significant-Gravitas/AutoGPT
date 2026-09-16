"use client";

import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { ExpertAbout } from "./components/ExpertAbout";
import { ExpertComingSoonLabel } from "./components/ExpertComingSoonLabel";
import { ExpertPageHeader } from "./components/ExpertPageHeader";
import { ExpertSkills } from "./components/ExpertSkills";
import { ExpertWorkflowList } from "./components/ExpertWorkflowList";
import { useExpertPage } from "./useExpertPage";

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

export default function MarketplaceExpertPage() {
  const { expertId } = useParams<{ expertId: string }>();
  const { expert, isLoading, isError, refetch } = useExpertPage({ expertId });

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

  return (
    <main className={MAIN_CLASS}>
      <BackToMarketplaceLink />
      <ExpertPageHeader
        expert={expert}
        accent={accent}
        actions={<ExpertComingSoonLabel />}
      />
      <div className="mt-8 flex flex-col gap-10 border-t border-zinc-200 pt-8">
        <ExpertAbout key={expert.id} text={expert.bio || expert.identity} />
        <ExpertSkills skills={expert.skills ?? []} accent={accent} />
        <ExpertWorkflowList
          name={expert.name}
          workflows={expert.workflows}
          accent={accent}
        />
      </div>
    </main>
  );
}
