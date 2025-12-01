import { Skeleton } from "@/components/__legacy__/ui/skeleton";

export const AgentPageLoading = () => {
  return (
    <div className="mx-auto w-full max-w-[1360px]">
      <main className="mt-5 px-4">
        <div className="flex items-center space-x-2">
          <Skeleton className="h-4 w-24" />
          <span>/</span>
          <Skeleton className="h-4 w-32" />
          <span>/</span>
          <Skeleton className="h-4 w-40" />
        </div>

        <div className="mt-8 flex flex-col gap-8 md:flex-row">
          <div className="w-full max-w-sm">
            <Skeleton className="h-64 w-full rounded-lg" />
          </div>
          <div className="flex-1">
            <Skeleton className="aspect-video w-full rounded-lg" />
          </div>
        </div>
      </main>
    </div>
  );
};
