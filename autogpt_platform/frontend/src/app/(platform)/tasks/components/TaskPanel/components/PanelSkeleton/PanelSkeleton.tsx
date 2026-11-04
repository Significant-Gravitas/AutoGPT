import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function PanelSkeleton() {
  return (
    <div
      aria-busy="true"
      aria-label="Loading task"
      className="flex flex-col gap-4 px-1.5 py-5"
    >
      <Skeleton className="h-7 w-3/4" />
      <div className="flex gap-2">
        <Skeleton className="h-7 w-24" />
        <Skeleton className="h-7 w-24" />
        <Skeleton className="h-7 w-24" />
      </div>
      <div className="flex flex-col gap-3 pt-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
        <Skeleton className="h-4 w-2/3" />
      </div>
      <Skeleton className="h-24 w-full" />
    </div>
  );
}
