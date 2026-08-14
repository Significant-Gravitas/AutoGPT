"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import dynamic from "next/dynamic";
import { notFound } from "next/navigation";

const RaiseFlow = dynamic(
  () =>
    import("./components/RaiseFlow/RaiseFlow").then(
      (module) => module.RaiseFlow,
    ),
  { ssr: false, loading: RaiseSkeleton },
);

export default function RaisePage() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);

  if (!ready) {
    return <RaiseSkeleton />;
  }
  if (!enabled) {
    notFound();
  }

  return <RaiseFlow />;
}

function RaiseSkeleton() {
  return (
    <main
      className="min-h-screen bg-zinc-50 px-4 pb-16 pt-6 sm:px-6 lg:px-8"
      aria-label="Loading…"
    >
      <div className="mx-auto grid w-full max-w-[1000px] gap-6 lg:grid-cols-[1fr_minmax(300px,360px)]">
        <div className="order-2 flex flex-col gap-4 lg:order-1">
          <Skeleton className="h-16 w-3/4 rounded-3xl" />
          <Skeleton className="h-40 w-full rounded-2xl" />
        </div>
        <div className="order-1 lg:order-2">
          <Skeleton className="h-80 w-full rounded-3xl" />
        </div>
      </div>
    </main>
  );
}
