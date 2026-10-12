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
      className="min-h-screen bg-background lg:h-screen lg:overflow-hidden"
      aria-label="Loading…"
    >
      <div className="grid w-full items-stretch lg:h-full lg:grid-cols-2">
        <div className="order-2 flex flex-col gap-4 px-4 pb-16 pt-6 sm:px-6 lg:order-1 lg:h-screen lg:overflow-y-auto lg:px-8">
          <Skeleton className="h-16 w-3/4 rounded-3xl" />
          <Skeleton className="h-40 w-full rounded-2xl" />
        </div>
        <div className="order-1 m-2 overflow-hidden rounded-[2.5rem] bg-muted/40 p-4 sm:p-6 lg:order-2">
          <Skeleton className="h-80 w-full rounded-[2rem]" />
        </div>
      </div>
    </main>
  );
}
