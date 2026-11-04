import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { InboxCard } from "./InboxCard";

const ROWS = ["a", "b", "c", "d"];

export function InboxListSkeleton() {
  return (
    <InboxCard>
      <div
        aria-busy="true"
        aria-label="Loading tasks"
        className="divide-y divide-border-hairline"
      >
        {ROWS.map((id) => (
          <div key={id} className="flex flex-col gap-2 py-3 pr-4 pl-4">
            <span className="flex items-center gap-3">
              <Skeleton className="h-4 max-w-md flex-1" />
              <Skeleton className="ml-auto hidden h-4 w-20 md:block" />
            </span>
            <span className="flex items-center gap-2">
              <Skeleton className="h-3.5 w-16" />
              <Skeleton className="h-3.5 w-48 max-w-[60%]" />
            </span>
          </div>
        ))}
      </div>
    </InboxCard>
  );
}
