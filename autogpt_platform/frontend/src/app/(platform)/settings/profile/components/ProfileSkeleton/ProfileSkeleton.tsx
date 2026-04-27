import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function ProfileSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-center sm:gap-6">
        <Skeleton className="h-[112px] w-[112px] rounded-full" />
        <div className="flex w-full max-w-[320px] flex-col gap-2">
          <Skeleton className="h-6 w-40" />
          <Skeleton className="h-4 w-56" />
          <Skeleton className="h-4 w-64" />
        </div>
      </div>
      <div className="flex flex-col gap-4 rounded-[16px] border border-zinc-200 bg-white p-6">
        <Skeleton className="h-5 w-28" />
        <Skeleton className="h-4 w-64" />
        <Skeleton className="h-[46px] w-full rounded-3xl" />
        <Skeleton className="h-[46px] w-full rounded-3xl" />
        <Skeleton className="h-[120px] w-full rounded-xl" />
      </div>
      <div className="flex flex-col gap-3 rounded-[16px] border border-zinc-200 bg-white p-6">
        <Skeleton className="h-5 w-28" />
        <Skeleton className="h-[46px] w-full rounded-3xl" />
        <Skeleton className="h-[46px] w-full rounded-3xl" />
      </div>
    </div>
  );
}
