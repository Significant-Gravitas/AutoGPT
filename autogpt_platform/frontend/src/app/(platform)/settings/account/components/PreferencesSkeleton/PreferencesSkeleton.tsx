import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function PreferencesSkeleton() {
  return (
    <div className="flex flex-col gap-6 pb-8" aria-busy="true">
      <div className="flex min-w-0 flex-col gap-3 pb-2 pl-4">
        <Skeleton className="h-7 w-32" />
        <Skeleton className="mt-2 h-4 w-[420px] max-w-full" />
      </div>

      <AccountSkeleton />
      <TimezoneSkeleton />
      <NotificationsSkeleton />

      <div className="flex items-center justify-end gap-2 px-4">
        <Skeleton className="h-12 w-32 rounded-full" />
        <Skeleton className="h-12 w-28 rounded-full" />
      </div>
    </div>
  );
}

function AccountSkeleton() {
  return (
    <div className="flex w-full flex-col gap-2">
      <div className="flex flex-col gap-1 px-4">
        <Skeleton className="h-5 w-24" />
        <Skeleton className="h-3.5 w-56" />
      </div>
      <div className="flex flex-col divide-y divide-zinc-200 rounded-[18px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex items-center justify-between gap-4 px-4 py-4">
          <Skeleton className="h-4 w-14" />
          <div className="flex items-center gap-3">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-7 w-7 rounded-full" />
          </div>
        </div>
        <div className="flex items-center justify-between gap-4 px-4 py-4">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-9 w-32 rounded-full" />
        </div>
      </div>
    </div>
  );
}

function TimezoneSkeleton() {
  return (
    <div className="flex w-full flex-col">
      <div className="flex flex-col gap-3 rounded-[18px] border border-zinc-200 bg-white px-4 py-3 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-4 w-4 rounded-full" />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton className="h-9 w-40 rounded-full" />
            <Skeleton className="h-7 w-20 rounded-full" />
          </div>
        </div>
      </div>
    </div>
  );
}

function NotificationsSkeleton() {
  return (
    <div className="flex w-full flex-col gap-2">
      <div className="flex flex-col gap-1 px-4">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-3.5 w-72" />
      </div>
      <div className="flex flex-col gap-3 rounded-[18px] border border-zinc-200 bg-white px-4 pb-4 pt-0 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex items-center gap-6 border-b border-zinc-100 pb-3 pt-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-5 w-28" />
          ))}
        </div>
        <div className="flex flex-col">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="flex items-center justify-between gap-4 rounded-[12px] px-4 py-3"
            >
              <div className="flex min-w-0 flex-col gap-2">
                <Skeleton className="h-4 w-56" />
                <Skeleton className="h-3 w-72" />
              </div>
              <Skeleton className="h-5 w-9 rounded-full" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
