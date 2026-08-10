"use client";

import { notFound } from "next/navigation";
import { getGreetingName } from "@/app/(platform)/copilot/components/EmptySession/helpers";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { AgentTeam } from "./components/AgentTeam/AgentTeam";
import { HomeHeader } from "./components/HomeHeader/HomeHeader";
import { MorningBriefing } from "./components/MorningBriefing/MorningBriefing";
import { NeedsYou } from "./components/NeedsYou/NeedsYou";
import { NowNext } from "./components/NowNext/NowNext";
import { getTimeOfDayGreeting } from "./helpers";
import { useHomePage } from "./useHomePage";

const SHELL_CLASS = "min-h-screen bg-zinc-50 px-4 pb-16 pt-5 sm:px-6 lg:px-8";
const CONTENT_CLASS = "mx-auto w-full max-w-[1180px]";
const GRID_CLASS = "grid grid-cols-1 items-start gap-7 xl:grid-cols-12";

export default function HomePage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const { user } = useAuth();
  const { dashboard, isLoading, isError, refetch } = useHomePage({
    enabled: Boolean(enabled) && ready,
  });

  if (!ready || isLoading) {
    return <HomeSkeleton />;
  }
  if (!enabled) {
    notFound();
  }
  if (isError || !dashboard) {
    return (
      <main className={SHELL_CLASS}>
        <div className={CONTENT_CLASS}>
          <ErrorCard
            context="home"
            httpError={{ message: "Your Home briefing could not be loaded" }}
            onRetry={() => refetch()}
          />
        </div>
      </main>
    );
  }

  return (
    <main className={SHELL_CLASS}>
      <div className={`${CONTENT_CLASS} flex flex-col gap-2.5`}>
        <HomeHeader
          greeting={getTimeOfDayGreeting()}
          name={getGreetingName(user)}
          dashboard={dashboard}
        />
        <div className={GRID_CLASS}>
          <div className="flex min-w-0 flex-col gap-7 xl:col-span-8">
            <NeedsYou dashboard={dashboard} />
            <MorningBriefing dashboard={dashboard} />
          </div>
          <div className="flex min-w-0 flex-col gap-7 xl:col-span-4">
            <AgentTeam dashboard={dashboard} />
            <NowNext dashboard={dashboard} />
          </div>
        </div>
      </div>
    </main>
  );
}

function HomeSkeleton() {
  return (
    <main className={SHELL_CLASS} aria-label="Loading Home…">
      <div className={`${CONTENT_CLASS} flex flex-col gap-2.5`}>
        <div className="my-6 flex items-start justify-between gap-6 px-1">
          <div className="space-y-2">
            <Skeleton className="h-9 w-72" />
            <Skeleton className="h-5 w-96 max-w-full" />
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1.5">
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-4 w-24" />
          </div>
        </div>
        <div className={GRID_CLASS}>
          <div className="flex flex-col gap-7 xl:col-span-8">
            <HomeTileSkeleton cardClassName="h-80" />
            <HomeTileSkeleton cardClassName="h-72" />
          </div>
          <div className="flex flex-col gap-7 xl:col-span-4">
            <HomeTileSkeleton cardClassName="h-56" />
            <HomeTileSkeleton cardClassName="h-72" />
          </div>
        </div>
      </div>
    </main>
  );
}

interface Props {
  cardClassName: string;
}

function HomeTileSkeleton({ cardClassName }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col justify-start space-y-1 px-4 sm:px-5">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-4 w-56 max-w-[70%]" />
      </div>
      <Skeleton className={`${cardClassName} rounded-[30px]`} />
    </div>
  );
}
