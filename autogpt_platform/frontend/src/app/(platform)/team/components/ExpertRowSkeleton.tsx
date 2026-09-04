import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ROW_CLASS } from "../helpers";

// Mirrors ExpertRow's two-line layout so the list doesn't jump when the real
// rows land.
export function ExpertRowSkeleton() {
  return (
    <div className={ROW_CLASS}>
      <Skeleton className="size-9 shrink-0 rounded-full" />
      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        <Skeleton className="h-3.5 w-28 rounded-full" />
        <Skeleton className="h-3 w-56 rounded-full" />
      </div>
      <Skeleton className="hidden h-3 w-20 shrink-0 rounded-full sm:block" />
    </div>
  );
}
