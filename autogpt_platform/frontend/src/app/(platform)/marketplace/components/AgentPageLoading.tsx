import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function AgentPageLoading() {
  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="mt-5 px-4 pb-12">
        {/* Breadcrumbs */}
        <div className="mb-4 flex items-center justify-between px-4 md:!-mb-3">
          <Skeleton className="h-8 w-20 rounded-lg" />
          <div className="hidden items-center gap-2 md:flex">
            <Skeleton className="h-4 w-24" />
            <span className="text-zinc-300">/</span>
            <Skeleton className="h-4 w-28" />
            <span className="text-zinc-300">/</span>
            <Skeleton className="h-4 w-36" />
          </div>
        </div>

        {/* Main content: Info left + Images right */}
        <div className="mt-0 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 lg:mt-8 lg:flex-row lg:gap-12">
          {/* Left: Agent info panel */}
          <div className="w-full lg:w-2/5">
            <div className="rounded-2xl bg-gradient-to-r from-blue-100/50 to-indigo-100/50 p-[1px]">
              <div className="flex flex-col rounded-[calc(1rem-2px)] bg-gray-50 p-4">
                {/* Title */}
                <Skeleton className="mb-3 h-9 w-3/4" />

                {/* Creator */}
                <div className="mb-3 flex items-center gap-2 lg:mb-12">
                  <Skeleton className="h-7 w-7 shrink-0 rounded-full" />
                  <Skeleton className="h-4 w-8" />
                  <Skeleton className="h-4 w-24" />
                </div>

                {/* Short description */}
                <Skeleton className="mb-4 h-5 w-full lg:mb-5" />
                <Skeleton className="mb-6 h-5 w-2/3" />

                {/* Buttons */}
                <div className="flex gap-3">
                  <Skeleton className="h-12 w-36 rounded-full" />
                  <Skeleton className="h-12 w-28 rounded-full" />
                </div>
              </div>
            </div>

            {/* Description section */}
            <div className="mt-8 space-y-3">
              <Skeleton className="h-6 w-28" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
            </div>

            {/* Categories */}
            <div className="mt-6 space-y-3">
              <Skeleton className="h-6 w-24" />
              <div className="flex flex-wrap gap-2">
                <Skeleton className="h-7 w-20 rounded-full" />
                <Skeleton className="h-7 w-24 rounded-full" />
                <Skeleton className="h-7 w-16 rounded-full" />
              </div>
            </div>
          </div>

          {/* Right: Image preview */}
          <div className="w-full px-2 lg:w-3/5 lg:flex-1">
            <Skeleton className="h-[15rem] w-full rounded-xl sm:h-[20rem] md:h-[25rem] lg:h-[30rem]" />

            {/* Thumbnails */}
            <div className="mt-3 flex gap-2 sm:mt-4 sm:gap-3">
              <Skeleton className="h-16 w-24 shrink-0 rounded-lg sm:h-20 sm:w-32" />
              <Skeleton className="h-16 w-24 shrink-0 rounded-lg sm:h-20 sm:w-32" />
              <Skeleton className="h-16 w-24 shrink-0 rounded-lg sm:h-20 sm:w-32" />
            </div>
          </div>
        </div>

        {/* Related agents section */}
        <div className="my-6" />
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Skeleton className="h-6 w-6" />
            <Skeleton className="h-6 w-48" />
          </div>
          <div className="hidden grid-cols-1 gap-6 md:grid md:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-[25rem] w-full rounded-2xl" />
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
