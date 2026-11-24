import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

const CARD_COUNT = 6;

// The list's own grid and the agent card's outline, so the page keeps its
// shape when the cards arrive.
export function LibraryAgentListSkeleton() {
  return (
    <div
      data-testid="library-agent-list-skeleton"
      role="status"
      aria-busy="true"
      aria-label="Loading workflows…"
      className="grid grid-cols-1 gap-6 sm:grid-cols-[repeat(auto-fill,minmax(21rem,1fr))]"
    >
      {Array.from({ length: CARD_COUNT }, (_, i) => (
        <div
          key={i}
          data-testid="library-agent-card-skeleton"
          className="flex min-h-[10.625rem] w-full max-w-[25rem] flex-col gap-2.5 rounded-medium border border-separator bg-surface"
        >
          <div className="px-4 pt-3">
            <Skeleton className="h-4 w-16" />
          </div>
          <div className="flex w-full flex-1 flex-col px-4 pb-2">
            <div className="flex w-full items-start justify-between gap-2">
              <div className="flex flex-1 flex-col gap-2">
                <Skeleton className="h-5 w-3/4" />
                <Skeleton className="h-5 w-1/2" />
              </div>
              <Skeleton className="h-[3.64rem] w-[6.70rem] flex-shrink-0" />
            </div>
            <div className="mt-4 flex w-full items-center justify-end gap-1 border-t border-separator pt-2">
              <Skeleton className="h-7 w-20" />
              <Skeleton className="h-7 w-20" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
