import { Skeleton } from "@/components/__legacy__/ui/skeleton";

export const MainMarketplacePageLoading = () => {
  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="px-4">
        <div className="flex flex-col gap-2 pt-16">
          <div className="flex flex-col items-center justify-center gap-8">
            <Skeleton className="h-16 w-[60%]" />
            <Skeleton className="h-12 w-[40%]" />
          </div>
          <div className="flex flex-col items-center justify-center gap-8 pt-8">
            <Skeleton className="h-8 w-[60%]" />
          </div>
          <div className="mx-auto flex w-[80%] flex-wrap items-center justify-center gap-8 pt-24">
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
            <Skeleton className="h-[12rem] w-[12rem]" />
          </div>
        </div>
      </main>
    </div>
  );
};
