import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function MainSearchResultPageLoading() {
  return (
    <div className="w-full">
      <div className="mx-auto min-h-screen max-w-[1440px] px-10 lg:min-w-[1440px]">
        {/* Go back button */}
        <div className="mb-4 mt-5">
          <Skeleton className="h-9 w-24 rounded-full" />
        </div>

        {/* Header: search term + search bar */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center">
          <div className="flex-1">
            <Skeleton className="mb-2 h-5 w-36" />
            <Skeleton className="h-8 w-56" />
          </div>
          <div className="flex-none">
            <Skeleton className="h-[2.75rem] w-full rounded-full md:w-[439px]" />
          </div>
        </div>

        {/* Filter chips + sort */}
        <div className="mt-6 flex flex-col gap-3 md:mt-9 md:flex-row md:items-center md:justify-between">
          <div className="flex gap-2">
            <Skeleton className="h-9 w-16 rounded-full" />
            <Skeleton className="h-9 w-20 rounded-full" />
            <Skeleton className="h-9 w-24 rounded-full" />
          </div>
          <Skeleton className="h-9 w-32 rounded-lg" />
        </div>

        {/* Agent cards grid */}
        <div className="space-y-8 py-8">
          <div className="hidden grid-cols-1 place-items-center gap-6 md:grid md:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-[25rem] w-full rounded-2xl" />
            ))}
          </div>
          {/* Mobile carousel placeholder */}
          <div className="flex gap-4 overflow-hidden md:hidden">
            <Skeleton className="h-[25rem] min-w-64 rounded-2xl" />
            <Skeleton className="h-[25rem] min-w-64 rounded-2xl" />
          </div>

          {/* Separator */}
          <Skeleton className="h-px w-full" />

          {/* Creator cards section */}
          <div className="space-y-6">
            <Skeleton className="h-6 w-24" />
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-[16rem] w-full rounded-2xl" />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
