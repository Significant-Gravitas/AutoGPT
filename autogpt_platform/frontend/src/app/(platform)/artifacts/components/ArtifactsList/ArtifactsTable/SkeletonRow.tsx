import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { ROW_GRID_CLASS } from "./row-layout";

export function SkeletonRow() {
  return (
    <div className={cn(ROW_GRID_CLASS, "px-2 py-2.5")}>
      <div className="flex items-center gap-4">
        <Skeleton className="h-10 w-10 rounded-xl" />
        <Skeleton className="h-4 w-48" />
      </div>
      <Skeleton className="hidden h-4 w-16 sm:block" />
      <Skeleton className="hidden h-4 w-12 md:block" />
      <span aria-hidden className="min-w-10" />
    </div>
  );
}
