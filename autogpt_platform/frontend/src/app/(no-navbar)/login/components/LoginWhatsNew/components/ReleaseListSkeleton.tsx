import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

// Detail lines per row, as the published releases usually wrap in the panel
// (every highlight, under a one-line title): the list keeps its height when
// they arrive.
const ROWS = [2, 2, 2];

// Matches a release row's line boxes.
export function ReleaseListSkeleton() {
  return (
    <div
      aria-hidden
      data-testid="whats-new-skeleton"
      className="flex flex-col gap-1 rounded-2xl bg-surface p-1.5 smooth-shadow-ring-sm"
    >
      {ROWS.map((detailLines, row) => (
        <div key={row} className="flex gap-3 px-3.5 py-2.5">
          <span className="flex w-2.5 shrink-0 justify-center pt-1">
            <span className="size-2 rounded-full bg-surface-hover" />
          </span>
          <div className="flex flex-1 flex-col">
            {/* The latest row's "Latest" pill makes its date line 18px. */}
            <div
              className={`flex items-center ${row === 0 ? "h-[1.125rem]" : "h-4"}`}
            >
              <Bar className="h-4 w-32" />
            </div>
            <div className="mt-1 flex h-[1.375rem] items-center">
              <Bar className="h-3.5 w-1/2" />
            </div>
            {detailLines === 2 ? (
              <div className="mt-0.5 flex h-10 flex-col justify-around">
                <Bar className="h-3 w-full" />
                <Bar className="h-3 w-2/3" />
              </div>
            ) : (
              <div className="mt-0.5 flex h-5 items-center">
                <Bar className="h-3 w-3/4" />
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function Bar({ className }: { className: string }) {
  return (
    <Skeleton
      className={`${className} rounded-full motion-reduce:animate-none`}
    />
  );
}
