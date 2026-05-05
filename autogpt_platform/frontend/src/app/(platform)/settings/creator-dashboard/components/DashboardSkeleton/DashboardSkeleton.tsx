import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-6 pb-8" aria-busy="true">
      <div className="flex flex-col gap-3 pl-4 pr-1 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-col gap-2">
          <Skeleton className="h-7 w-48" />
          <Skeleton className="mt-1 h-4 w-[420px] max-w-full" />
        </div>
        <Skeleton className="h-12 w-36 rounded-full" />
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="relative flex min-h-[132px] flex-col justify-between gap-3 rounded-[18px] border border-zinc-200 bg-white px-4 py-5"
          >
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-7 w-16" />
            <Skeleton className="absolute right-3 top-3 h-7 w-7 rounded-full" />
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-4 border-b border-zinc-100 px-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="my-2 h-6 w-20" />
        ))}
      </div>

      <div className="overflow-hidden rounded-[18px] border border-zinc-200 bg-white">
        <div className="border-b border-zinc-100 bg-zinc-50/60 px-4 py-3">
          <Skeleton className="h-4 w-24" />
        </div>
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-3 border-b border-zinc-100 px-4 py-3 last:border-b-0"
          >
            <Skeleton className="aspect-video h-12 w-20 shrink-0 rounded-[8px]" />
            <div className="flex flex-1 flex-col gap-2">
              <Skeleton className="h-4 w-1/3" />
              <Skeleton className="h-3 w-2/3" />
            </div>
            <Skeleton className="h-6 w-24 rounded-full" />
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-4 w-12" />
            <Skeleton className="h-4 w-12" />
            <Skeleton className="h-8 w-8 rounded-full" />
          </div>
        ))}
      </div>
    </div>
  );
}
