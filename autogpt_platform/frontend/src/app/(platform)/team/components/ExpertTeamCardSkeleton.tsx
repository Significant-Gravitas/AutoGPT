import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

// Mirrors ExpertTeamCard's structure — banner, straddling avatar, tagline,
// credits meter, footer actions — so the grid doesn't jump when the real
// cards land. A single flat block would settle at the wrong height.
export function ExpertTeamCardSkeleton() {
  return (
    <div className="flex flex-col rounded-[1.75rem] border border-zinc-200 bg-white p-1">
      <Skeleton className="h-24 w-full rounded-t-[1.5rem]" />

      <div className="flex flex-1 flex-col gap-3 p-4 pt-0">
        <div className="flex items-center gap-3">
          <Skeleton className="-mt-9 size-[4.5rem] shrink-0 self-start rounded-full ring-4 ring-white" />
          <div className="flex min-w-0 flex-1 flex-col gap-2">
            <Skeleton className="h-4 w-2/5 rounded-full" />
            <Skeleton className="h-3 w-1/4 rounded-full" />
          </div>
        </div>

        <div className="flex min-h-12 flex-col gap-2">
          <Skeleton className="h-3 w-full rounded-full" />
          <Skeleton className="h-3 w-4/5 rounded-full" />
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex items-baseline justify-between gap-2">
            <Skeleton className="h-3 w-28 rounded-full" />
            <Skeleton className="h-3 w-16 rounded-full" />
          </div>
          <Skeleton className="h-4 w-full rounded-sm" />
        </div>

        <Skeleton className="h-3 min-h-5 w-32 rounded-full" />
        <Skeleton className="h-3 min-h-5 w-24 rounded-full" />

        <div className="mt-auto flex gap-2">
          <Skeleton className="h-9 w-[7.7rem] rounded-full" />
          <Skeleton className="h-9 w-[9.5rem] rounded-full" />
        </div>
      </div>
    </div>
  );
}
