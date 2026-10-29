import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { DATE_CELL_CLASS, ROW_GRID_CLASS, SIZE_CELL_CLASS } from "./row-layout";

export function SkeletonRow() {
  return (
    <div className={cn(ROW_GRID_CLASS, "px-2 py-2.5")}>
      <div className="flex items-center gap-4">
        <Skeleton className="h-10 w-10 rounded-xl" />
        <Skeleton className="h-4 w-48" />
      </div>
      <Skeleton className={cn(DATE_CELL_CLASS, "h-4 w-16")} />
      <Skeleton className={cn(SIZE_CELL_CLASS, "h-4 w-12")} />
      <span aria-hidden className="min-w-10" />
    </div>
  );
}
