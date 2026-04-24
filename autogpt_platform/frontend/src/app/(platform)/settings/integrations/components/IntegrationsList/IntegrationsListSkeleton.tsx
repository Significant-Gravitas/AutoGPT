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
          className="w-full overflow-hidden rounded-lg border border-[#DADADC] bg-white"
        >
          <div className="flex items-center gap-3 px-3 py-3 pr-5">
            <Skeleton className="size-6 rounded-full" />
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-5 w-8 rounded-full" />
          </div>
        </div>
      ))}
    </div>
  );
}
