import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

// Mirrors ExpertTeamCard — cover, overlapping avatar, centred name and role,
// stats pill, spend meter, icon footer — so the grid doesn't jump when the
// real cards land. A single flat block would settle at the wrong height.
export function ExpertTeamCardSkeleton() {
  return (
    <div className="flex flex-col overflow-hidden rounded-[2.25rem] bg-white shadow-zinc-950 smooth-shadow-ring-sm">
      <div className="flex flex-col items-start p-3 pb-5">
        <div className="h-32 w-full rounded-[1.75rem] bg-zinc-100" />
        <div className="flex w-full items-start gap-3 px-2">
          <Skeleton className="-mt-11 ml-1 size-[5.75rem] shrink-0 rounded-full" />
          <div className="mt-2 flex flex-1 flex-col gap-1">
            <div className="flex items-baseline justify-between gap-2">
              <Skeleton className="h-3 w-12 rounded-full" />
              <Skeleton className="h-3 w-14 rounded-full" />
            </div>
            <Skeleton className="h-4 w-full rounded-sm" />
          </div>
        </div>
        <div className="mt-2 flex w-full flex-col items-start gap-2 pl-5 pr-2">
          <Skeleton className="h-5 w-2/5 rounded-full" />
          <Skeleton className="h-4 w-1/2 rounded-full" />
        </div>

        <div className="mx-2 mt-4 flex rounded-3xl bg-zinc-50 px-2 py-3 ring-1 ring-inset ring-zinc-100">
          {[0, 1, 2, 3].map((stat) => (
            <div
              key={stat}
              className="flex flex-1 flex-col items-center gap-1.5"
            >
              <Skeleton className="h-5 w-7 rounded-full" />
              <Skeleton className="h-3 w-12 rounded-full" />
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-2 px-5 pb-5">
        <Skeleton className="h-9 flex-1 rounded-2xl" />
        <Skeleton className="h-9 flex-1 rounded-2xl" />
      </div>
    </div>
  );
}
