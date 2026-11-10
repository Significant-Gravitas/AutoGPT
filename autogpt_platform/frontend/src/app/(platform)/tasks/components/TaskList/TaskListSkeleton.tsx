import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { TasksCard } from "./TasksCard";

const ROWS = ["a", "b", "c", "d"];

export function TaskListSkeleton() {
  return (
    <TasksCard>
      <div
        aria-busy="true"
        aria-label="Loading tasks"
        className="divide-y divide-border-hairline"
      >
        {ROWS.map((id) => (
          <div key={id} className="flex items-center gap-3 py-3 pr-4 pl-4">
            <Skeleton className="size-4 rounded-full" />
            <Skeleton className="h-4 max-w-md flex-1" />
            <Skeleton className="ml-auto hidden h-4 w-20 md:block" />
            <Skeleton className="h-4 w-10" />
          </div>
        ))}
      </div>
    </TasksCard>
  );
}
