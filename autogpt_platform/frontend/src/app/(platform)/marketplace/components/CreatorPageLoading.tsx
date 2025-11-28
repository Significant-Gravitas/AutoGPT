import { Skeleton } from "@/components/__legacy__/ui/skeleton";

export const CreatorPageLoading = () => {
  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="mt-5 px-4">
        <Skeleton className="mb-4 h-6 w-40" />

        <div className="mt-4 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 md:mt-8 md:flex-row md:gap-8">
          <div className="w-full md:w-auto md:shrink-0">
            <Skeleton className="h-80 w-80 rounded-xl" />
            <div className="mt-4 space-y-2">
              <Skeleton className="h-6 w-80" />
              <Skeleton className="h-4 w-80" />
            </div>
          </div>

          <div className="flex min-w-0 flex-1 flex-col gap-4">
            <Skeleton className="h-6 w-24" />
            <Skeleton className="h-8 w-full max-w-xl" />
            <Skeleton className="h-4 w-1/2" />
            <div className="flex gap-2">
              <Skeleton className="h-8 w-8 rounded-full" />
              <Skeleton className="h-8 w-8 rounded-full" />
            </div>
          </div>
        </div>

        <div className="mt-8">
          <Skeleton className="mb-6 h-px w-full" />
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-32 w-full rounded-lg" />
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};
