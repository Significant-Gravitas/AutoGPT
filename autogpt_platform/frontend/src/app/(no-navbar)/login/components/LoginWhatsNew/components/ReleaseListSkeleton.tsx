import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import {
  DATE_LINE_HEIGHT_CLASS,
  DETAILS_LINE_HEIGHT_CLASS,
  HEADLINE_LINE_HEIGHT_CLASS,
  LATEST_DATE_LINE_HEIGHT_CLASS,
  RELEASE_DETAILS_GAP_CLASS,
  RELEASE_DOT_COLUMN_CLASS,
  RELEASE_HEADLINE_GAP_CLASS,
  RELEASE_LIST_CLASS,
  RELEASE_ROW_CLASS,
} from "./releaseRowLayout";

// Detail lines per row, as the published releases usually wrap in the panel
// (every highlight, under a one-line title).
const ROWS = [2, 2, 2];

// Built from the release row's own layout, line box for line box.
export function ReleaseListSkeleton() {
  return (
    <div
      aria-hidden
      data-testid="whats-new-skeleton"
      className={RELEASE_LIST_CLASS}
    >
      {ROWS.map((detailLines, row) => (
        <div key={row} className={RELEASE_ROW_CLASS}>
          <span className={RELEASE_DOT_COLUMN_CLASS}>
            <span className="size-2 rounded-full bg-surface-hover" />
          </span>
          <div className="flex flex-1 flex-col">
            <div
              className={cn(
                "flex items-center",
                row === 0
                  ? LATEST_DATE_LINE_HEIGHT_CLASS
                  : DATE_LINE_HEIGHT_CLASS,
              )}
            >
              <Bar className="h-4 w-32" />
            </div>
            <div
              className={cn(
                RELEASE_HEADLINE_GAP_CLASS,
                HEADLINE_LINE_HEIGHT_CLASS,
                "flex items-center",
              )}
            >
              <Bar className="h-3.5 w-1/2" />
            </div>
            <div className={cn(RELEASE_DETAILS_GAP_CLASS, "flex flex-col")}>
              {Array.from({ length: detailLines }, (_, line) => (
                <div
                  key={line}
                  className={cn(DETAILS_LINE_HEIGHT_CLASS, "flex items-center")}
                >
                  <Bar
                    className={cn(
                      "h-3",
                      line === detailLines - 1 ? "w-2/3" : "w-full",
                    )}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function Bar({ className }: { className: string }) {
  return (
    <Skeleton
      className={cn(className, "rounded-full motion-reduce:animate-none")}
    />
  );
}
