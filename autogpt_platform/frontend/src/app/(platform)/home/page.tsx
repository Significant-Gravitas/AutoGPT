"use client";

import { notFound } from "next/navigation";
import { getGreetingName } from "@/app/(platform)/copilot/components/EmptySession/helpers";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { AgentTeam } from "./components/AgentTeam/AgentTeam";
import { HomeBackdrop } from "./components/HomeBackdrop/HomeBackdrop";
import { HomeHeader } from "./components/HomeHeader/HomeHeader";
import { NeedsYou } from "./components/NeedsYou/NeedsYou";
import { NowNext } from "./components/NowNext/NowNext";
import { RecentWork } from "./components/RecentWork/RecentWork";
import { getTimeOfDayGreeting } from "./helpers";
import { useHomePage } from "./useHomePage";

const SHELL_CLASS =
  "relative min-h-screen bg-zinc-50 px-4 pb-16 pt-6 sm:px-6 lg:px-8";
const CONTENT_CLASS = "relative mx-auto w-full max-w-[1120px]";
const GRID_CLASS = "grid grid-cols-1 items-start gap-4 xl:grid-cols-12";

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
        <HomeBackdrop />
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
      <HomeBackdrop />
      <div className={CONTENT_CLASS}>
        <HomeHeader
          greeting={getTimeOfDayGreeting()}
          name={getGreetingName(user)}
          dashboard={dashboard}
        />
        <div className={GRID_CLASS}>
          <div className="flex min-w-0 flex-col gap-4 xl:col-span-8">
            {/* An empty inbox is not news: the header already says nothing
                needs you, so the tile only appears once there is something
                to decide. */}
            {dashboard.attention.length > 0 && (
              <NeedsYou dashboard={dashboard} />
            )}
            <RecentWork dashboard={dashboard} />
          </div>
          <div className="flex min-w-0 flex-col gap-4 xl:col-span-4">
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
      <HomeBackdrop />
      <div className={CONTENT_CLASS}>
        <div className="flex items-end justify-between gap-6 px-1 pb-5 pt-1">
          <div className="space-y-2">
            <Skeleton className="h-7 w-64" />
            <Skeleton className="h-4 w-80 max-w-full" />
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1.5">
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-3 w-24" />
          </div>
        </div>
        <div className={GRID_CLASS}>
          <div className="flex flex-col gap-4 xl:col-span-8">
            <HomeTileSkeleton cardClassName="h-64" />
            <HomeTileSkeleton cardClassName="h-56" />
          </div>
          <div className="flex flex-col gap-4 xl:col-span-4">
            <HomeTileSkeleton cardClassName="h-44" />
            <HomeTileSkeleton cardClassName="h-56" />
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
    <div className="overflow-hidden rounded-lg border border-zinc-200 bg-white">
      <div className="flex h-10 items-center border-b border-zinc-100 px-4">
        <Skeleton className="h-3.5 w-28" />
      </div>
      <Skeleton className={`${cardClassName} rounded-none`} />
    </div>
  );
}
