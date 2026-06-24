import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

interface Props {
  /** Number of skeleton rows to render. Defaults to 4 — roughly the
   * average bucket size in the global search modal. */
  rows?: number;
}

const STAGGER_MS = 35;

export function SearchCommandSkeleton({ rows = 4 }: Props) {
  return (
    <div
      role="presentation"
      aria-hidden="true"
      data-testid="search-command-skeleton"
      className="flex flex-col gap-1 px-2"
    >
      <div className="px-3 pb-1 pt-2">
        <Skeleton className="h-3 w-16" />
      </div>
      <div className="flex flex-col">
        {Array.from({ length: rows }, (_, idx) => (
          <div
            key={idx}
            className="flex items-center gap-2.5 rounded-md px-3 py-2 motion-safe:animate-search-item-in"
            style={{ animationDelay: `${idx * STAGGER_MS}ms` }}
          >
            <Skeleton className="h-4 w-4 rounded" />
            <div className="flex min-w-0 flex-1 flex-col gap-1">
              <Skeleton className="h-3.5 w-[55%]" />
              <Skeleton className="h-3 w-[35%]" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
