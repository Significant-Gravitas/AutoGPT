"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

const PLACEHOLDER_ROWS = Array.from({ length: 6 }, (_, i) => i);

export function APIKeyListSkeleton() {
  return (
    <div
      role="status"
      aria-label="Loading API keys"
      className="flex w-full flex-col divide-y divide-zinc-200 overflow-hidden rounded-[8px] border border-zinc-200 bg-white"
    >
      {PLACEHOLDER_ROWS.map((i) => (
        <div
          key={i}
          className="flex items-center justify-between py-4 pl-3 pr-5"
        >
          <div className="flex items-center gap-3">
            <Skeleton className="h-5 w-5" />
            <div className="flex flex-col gap-2">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-3 w-56" />
            </div>
          </div>
          <Skeleton className="h-5 w-5" />
        </div>
      ))}
    </div>
  );
}
