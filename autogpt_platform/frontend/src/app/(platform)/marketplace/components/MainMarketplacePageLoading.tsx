import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

export function MainMarketplacePageLoading() {
  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="px-4">
        <div className="flex flex-col gap-2 pt-16">
          <div className="flex flex-col items-center justify-center gap-8">
            <Skeleton className="h-16 w-3/5" />
            <Skeleton className="h-12 w-2/5" />
          </div>
          <div className="flex flex-col items-center justify-center gap-8 pt-8">
            <Skeleton className="h-8 w-3/5" />
          </div>
          <div className="mx-auto flex w-4/5 flex-wrap items-center justify-center gap-8 pt-24">
            {Array.from({ length: 10 }).map((_, index) => (
              <Skeleton key={index} className="size-48" />
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
