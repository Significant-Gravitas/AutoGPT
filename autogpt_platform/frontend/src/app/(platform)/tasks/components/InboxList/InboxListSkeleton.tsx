import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { InboxTray } from "./InboxTray";

const ROWS = ["a", "b", "c", "d"];

export function InboxListSkeleton() {
  return (
    <InboxTray>
      <div
        aria-busy="true"
        aria-label="Loading tasks"
        className="divide-y divide-border-hairline"
      >
        {ROWS.map((id) => (
          <div key={id} className="flex items-center gap-3 px-4 py-3.5">
            <Skeleton className="size-4 rounded-full" />
            <Skeleton className="h-4 flex-1" />
            <Skeleton className="h-4 w-16" />
          </div>
        ))}
      </div>
    </InboxTray>
  );
}
