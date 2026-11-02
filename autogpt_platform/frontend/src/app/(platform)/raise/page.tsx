"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import dynamic from "next/dynamic";

const RaiseFlow = dynamic(
  () =>
    import("./components/RaiseFlow/RaiseFlow").then(
      (module) => module.RaiseFlow,
    ),
  { ssr: false, loading: RaiseSkeleton },
);

export default function RaisePage() {
  return <RaiseFlow />;
}

function RaiseSkeleton() {
  return (
    <main
      className="min-h-screen bg-background lg:h-screen lg:overflow-hidden"
      aria-label="Loading…"
    >
      <div className="grid w-full items-stretch lg:h-full lg:grid-cols-2">
        <div className="order-2 flex flex-col gap-4 px-4 pt-6 pb-16 sm:px-6 lg:order-1 lg:h-screen lg:overflow-y-auto lg:px-8">
          <Skeleton className="h-16 w-3/4 rounded-3xl" />
          <Skeleton className="h-40 w-full rounded-2xl" />
        </div>
        {/* oxlint-disable-next-line design-system/no-arbitrary-values -- the 40px panel radius has no rounded-* step (4xl is 32px) */}
        <div className="order-1 m-2 overflow-hidden rounded-[2.5rem] bg-muted/40 p-4 sm:p-6 lg:order-2">
          {/* oxlint-disable-next-line design-system/no-arbitrary-values -- the 32px radius has no rounded-* step in this theme (3xl is 24px) */}
          <Skeleton className="h-80 w-full rounded-[2rem]" />
        </div>
      </div>
    </main>
  );
}
