import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Separator } from "@radix-ui/react-separator";

export function ProfileLoading() {
  return (
    <div className="flex flex-col items-center justify-center px-4">
      <div className="w-full min-w-[800px] px-4 sm:px-8">
        <Skeleton className="mb-6 h-[35px] w-32 sm:mb-8" />
        <div className="mb-8 sm:mb-12">
          <div className="mb-8 flex flex-col items-center gap-4 sm:flex-row sm:items-start">
            <Skeleton className="h-[130px] w-[130px] rounded-full" />
            <Skeleton className="mt-11 h-[43px] w-32 rounded-[22px]" />
          </div>
          <div className="space-y-4 sm:space-y-6">
            <div className="w-full space-y-2">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-10 w-full rounded-[55px]" />
            </div>
            <div className="w-full space-y-2">
              <Skeleton className="h-4 w-16" />
              <Skeleton className="h-10 w-full rounded-[55px]" />
            </div>
            <div className="w-full space-y-2">
              <Skeleton className="h-4 w-12" />
              <Skeleton className="h-[220px] w-full rounded-2xl" />
            </div>
            <section className="mb-8">
              <Skeleton className="mb-4 h-6 w-32" />
              <Skeleton className="mb-6 h-4 w-64" />
              <div className="space-y-4 sm:space-y-6">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="w-full space-y-2">
                    <Skeleton className="h-4 w-16" />
                    <Skeleton className="h-10 w-full rounded-[55px]" />
                  </div>
                ))}
              </div>
            </section>
            <Separator />
            <div className="flex h-[50px] items-center justify-end gap-3 py-8">
              <Skeleton className="h-[50px] w-32 rounded-[35px]" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
