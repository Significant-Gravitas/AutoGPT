import React from "react";
import { Skeleton } from "@/components/ui/skeleton";

export default function LibraryRunLoadingSkeleton() {
  return (
    <div className="container justify-stretch p-0 lg:flex">
      {/* Skeleton for sidebar with runs list */}
      <div className="agpt-div w-[400px] border-b lg:border-b-0 lg:border-r">
        <div className="p-4">
          <Skeleton className="h-16 w-3/4" />
          <div className="mt-4 space-y-2">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-center space-x-2 py-2">
                <Skeleton className="h-8 w-full" />
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1">
        {/* Skeleton for agent header */}
        <div className="agpt-div w-full border-b p-4">
          <Skeleton className="h-8 w-1/2" />
        </div>

        {/* Skeleton for run details */}
        <div className="p-4">
          <div className="flex justify-between">
            <Skeleton className="h-6 w-1/3" />
            <Skeleton className="h-6 w-1/4" />
          </div>
          <div className="mt-6 space-y-4">
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-60 w-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
