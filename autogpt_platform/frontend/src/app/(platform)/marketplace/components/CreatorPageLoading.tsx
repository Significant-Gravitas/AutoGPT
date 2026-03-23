import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function CreatorPageLoading() {
  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="mt-5 px-4 pb-12">
        {/* Breadcrumbs */}
        <div className="mb-4 flex items-center justify-between px-4 md:!-mb-3">
          <Skeleton className="h-8 w-20 rounded-lg" />
          <div className="hidden items-center gap-2 md:flex">
            <Skeleton className="h-4 w-24" />
            <span className="text-zinc-300">/</span>
            <Skeleton className="h-4 w-32" />
          </div>
        </div>

        {/* Main content */}
        <div className="mt-0 flex flex-col items-start gap-4 sm:mt-6 sm:gap-6 lg:mt-8 lg:flex-row lg:gap-12">
          {/* Left: Creator info card */}
          <div className="w-full lg:w-2/5">
            <div className="w-full px-4 sm:px-6 lg:px-0">
              <div className="rounded-2xl bg-gradient-to-r from-blue-100/50 to-indigo-100/50 p-[1px]">
                <div className="flex flex-col rounded-[calc(1rem-2px)] bg-gray-50 p-4">
                  {/* Avatar */}
                  <Skeleton className="mb-4 h-20 w-20 rounded-full sm:h-24 sm:w-24" />

                  {/* Name */}
                  <Skeleton className="mb-1 h-9 w-48" />

                  {/* Handle */}
                  <Skeleton className="mb-4 h-5 w-28" />

                  {/* Description */}
                  <div className="mb-6 space-y-2">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                  </div>

                  {/* Categories */}
                  <div className="mb-6">
                    <Skeleton className="mb-2 h-5 w-28" />
                    <div className="flex flex-wrap gap-2">
                      <Skeleton className="h-7 w-20 rounded-full" />
                      <Skeleton className="h-7 w-24 rounded-full" />
                      <Skeleton className="h-7 w-16 rounded-full" />
                    </div>
                  </div>

                  {/* Links */}
                  <Skeleton className="mb-2 h-5 w-14" />
                  <div className="flex flex-wrap gap-2">
                    <Skeleton className="h-9 w-28 rounded-full" />
                    <Skeleton className="h-9 w-32 rounded-full" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right side spacer */}
          <div className="hidden lg:block lg:w-3/5" />
        </div>

        {/* Agent grid section */}
        <div className="my-6" />
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Skeleton className="h-6 w-6" />
            <Skeleton className="h-6 w-40" />
          </div>
          <div className="hidden grid-cols-1 gap-6 md:grid md:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-[25rem] w-full rounded-2xl" />
            ))}
          </div>
          {/* Mobile carousel placeholder */}
          <div className="flex gap-4 overflow-hidden md:hidden">
            <Skeleton className="h-[25rem] min-w-64 rounded-2xl" />
            <Skeleton className="h-[25rem] min-w-64 rounded-2xl" />
          </div>
        </div>
      </main>
    </div>
  );
}
