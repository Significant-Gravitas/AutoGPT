import { Skeleton } from "@/components/__legacy__/ui/skeleton";

export const MainSearchResultPageLoading = () => {
  return (
    <div className="w-full">
      <div className="mx-auto min-h-screen max-w-[1440px] px-10 lg:min-w-[1440px]">
        <div className="mt-8 flex items-center">
          <div className="flex-1">
            <Skeleton className="mb-2 h-5 w-32 bg-neutral-200 dark:bg-neutral-700" />
            <Skeleton className="h-8 w-64 bg-neutral-200 dark:bg-neutral-700" />
          </div>
          <div className="flex-none">
            <Skeleton className="h-[60px] w-[439px] bg-neutral-200 dark:bg-neutral-700" />
          </div>
        </div>
        <div className="mt-[36px] flex items-center justify-between">
          <Skeleton className="h-8 w-48 bg-neutral-200 dark:bg-neutral-700" />
          <Skeleton className="h-8 w-32 bg-neutral-200 dark:bg-neutral-700" />
        </div>
        <div className="mt-20 flex flex-col items-center justify-center">
          <Skeleton className="mb-4 h-6 w-40 bg-neutral-200 dark:bg-neutral-700" />
          <Skeleton className="h-6 w-80 bg-neutral-200 dark:bg-neutral-700" />
        </div>
      </div>
    </div>
  );
};
