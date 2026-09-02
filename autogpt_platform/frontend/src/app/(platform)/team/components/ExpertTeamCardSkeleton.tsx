import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

// Mirrors ExpertTeamCard's structure — stacked avatar, name and role, spend
// meter, footer actions — so the grid doesn't jump when the real cards land.
// A single flat block would settle at the wrong height.
export function ExpertTeamCardSkeleton() {
  return (
    <div className="flex flex-col rounded-[1.75rem] bg-white p-1 shadow-zinc-950 smooth-shadow-ring-sm">
      <div className="flex flex-1 flex-col gap-3 p-4">
        <div className="flex flex-col gap-2">
          <Skeleton className="size-14 shrink-0 self-start rounded-2xl" />
          <div className="flex flex-col gap-2">
            <Skeleton className="h-4 w-2/5 rounded-full" />
            <Skeleton className="h-3 w-1/4 rounded-full" />
          </div>
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
          <Skeleton className="h-9 w-[7.7rem] rounded-xl" />
          <Skeleton className="h-9 w-[9.5rem] rounded-xl" />
        </div>
      </div>
    </div>
  );
}
