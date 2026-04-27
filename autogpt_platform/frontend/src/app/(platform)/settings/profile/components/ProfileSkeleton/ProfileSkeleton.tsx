import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function ProfileSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col pb-2">
        <Skeleton className="h-7 w-32" />
        <Skeleton className="mt-4 h-4 w-72" />
      </div>

      <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-start sm:gap-6">
        <Skeleton className="h-[112px] w-[112px] shrink-0 rounded-full" />
        <div className="grid w-full gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-2">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-[46px] w-full rounded-3xl" />
          </div>
          <div className="flex flex-col gap-2">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-[46px] w-full rounded-3xl" />
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-12" />
        <Skeleton className="h-[120px] w-full rounded-3xl" />
      </div>

      <div className="flex flex-col gap-3">
        <Skeleton className="h-4 w-24" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Skeleton className="h-[46px] w-full rounded-3xl" />
          <Skeleton className="h-[46px] w-full rounded-3xl" />
          <Skeleton className="h-[46px] w-full rounded-3xl" />
        </div>
      </div>
    </div>
  );
}
