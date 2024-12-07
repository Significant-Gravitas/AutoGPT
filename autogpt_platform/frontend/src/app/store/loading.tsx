import React from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";

export default function StoreLoadingSkeleton() {
  return (
    <div className="mx-auto w-screen max-w-[1360px]">
      <main className="flex flex-col gap-12 px-4 pt-8">
        {/* Hero Section Skeleton */}
        <div className="my-8 flex flex-col items-center">
          <Skeleton className="mb-4 h-12 w-3/4" />
          <Skeleton className="mb-8 h-6 w-1/2" />
        </div>

        {/* Featured Section Skeleton */}
        <div className="py-8">
          <Skeleton className="mb-6 h-8 w-1/4" />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-3">
                <Skeleton className="h-48 w-full" />
                <Skeleton className="h-6 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            ))}
          </div>
        </div>

        <Separator />

        {/* Top Agents Section Skeleton */}
        <div className="py-8">
          <Skeleton className="mb-6 h-8 w-1/4" />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="space-y-3">
                <Skeleton className="h-40 w-full" />
                <Skeleton className="h-5 w-2/3" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            ))}
          </div>
        </div>

        <Separator />

        {/* Featured Creators Section Skeleton */}
        <div className="py-8">
          <Skeleton className="mb-6 h-8 w-1/4" />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-3">
                <Skeleton className="h-16 w-16 rounded-full" />
                <Skeleton className="h-5 w-1/2" />
                <Skeleton className="h-4 w-3/4" />
              </div>
            ))}
          </div>
        </div>

        <Separator />

        {/* Become Creator Section Skeleton */}
        <div className="py-8 text-center">
          <Skeleton className="mx-auto mb-4 h-8 w-1/3" />
          <Skeleton className="mx-auto mb-6 h-4 w-1/2" />
          <Skeleton className="mx-auto h-10 w-40" />
        </div>
      </main>
    </div>
  );
}
