import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function IntegrationsListSkeleton() {
  return (
    <div
      className="flex flex-col gap-3"
      aria-busy="true"
      aria-label="Loading integrations"
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          data-testid="integration-skeleton-item"
          className="w-full overflow-hidden rounded-lg border border-[#DADADC] bg-white"
        >
          {/* Mirrors ProviderGroup accordion trigger row */}
          <div className="flex items-center justify-between px-3 py-3 pr-5">
            <div className="flex items-center gap-3">
              <Skeleton className="size-6 rounded-full" />
              <Skeleton className="h-[22px] w-32" />
              <Skeleton className="h-[22px] w-8 rounded-[10px]" />
            </div>
            <Skeleton className="size-4 rounded" />
          </div>
          {/* Mirrors first credential row inside accordion content */}
          <div className="flex items-center justify-between border-t border-[#DADADC] py-3 pl-3 pr-5">
            <div className="flex items-center gap-3">
              <Skeleton className="size-5 rounded" />
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-[22px] w-40" />
                  <Skeleton className="h-[20px] w-14 rounded-[10px]" />
                </div>
                <Skeleton className="h-3 w-28" />
              </div>
            </div>
            <Skeleton className="size-5 rounded" />
          </div>
        </div>
      ))}
    </div>
  );
}
