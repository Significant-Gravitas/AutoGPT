import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function PreferencesSkeleton() {
  return (
    <div className="flex flex-col gap-6 pb-28" aria-busy="true">
      <div className="flex flex-col gap-3">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-4 w-72" />
      </div>

      <SectionSkeleton rows={2} />
      <SectionSkeleton rows={1} />
      <SectionSkeleton rows={6} />
    </div>
  );
}

function SectionSkeleton({ rows }: { rows: number }) {
  return (
    <div className="flex flex-col gap-4 rounded-[16px] border border-zinc-200 bg-white p-6">
      <div className="flex flex-col gap-2">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-3.5 w-56" />
      </div>
      <div className="flex flex-col gap-3">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex items-center justify-between gap-4">
            <div className="flex flex-col gap-2">
              <Skeleton className="h-4 w-44" />
              <Skeleton className="h-3 w-64" />
            </div>
            <Skeleton className="h-5 w-9 rounded-full" />
          </div>
        ))}
      </div>
    </div>
  );
}
