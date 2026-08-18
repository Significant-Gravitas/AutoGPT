"use client";

import { useGetWorkspaceStorageUsage } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { formatFileSize } from "../ArtifactsList/helpers";

const SEGMENT_COUNT = 40;

export function StorageUsage() {
  const { data, isLoading, isError } = useGetWorkspaceStorageUsage({
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
    },
  });

  if (isLoading) {
    return (
      <div
        className="flex w-full flex-col gap-2"
        data-testid="storage-usage-loading"
      >
        <Skeleton className="h-3 w-24" />
        <div className="flex items-center gap-3">
          <Skeleton className="h-8 flex-1 rounded-md" />
          <Skeleton className="h-3 w-32" />
        </div>
      </div>
    );
  }

  if (isError || !data) return null;

  const percent = Math.min(Math.max(data.used_percent, 0), 100);
  const filled = Math.round((percent / 100) * SEGMENT_COUNT);
  const filledColor = percent > 95 ? "bg-red-400" : "bg-purple-300";
  const usedLabel = `${Math.round(percent)}% used`;
  const leftLabel = `${Math.max(0, Math.round(100 - percent))}% left`;

  return (
    <div className="flex w-full flex-col gap-2" data-testid="storage-usage">
      <Text variant="body-medium" className="text-zinc-800">
        Storage
      </Text>
      <div className="flex items-center gap-3">
        <TooltipProvider delayDuration={400} skipDelayDuration={120}>
          <div
            role="progressbar"
            aria-valuenow={percent}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`${usedLabel}, ${leftLabel}`}
            className="flex h-8 flex-1 items-center gap-[3px]"
          >
            {Array.from({ length: SEGMENT_COUNT }).map((_, i) => {
              const isFilled = i < filled;
              return (
                <Tooltip key={i} delayDuration={400}>
                  <TooltipTrigger asChild>
                    <div
                      className={cn(
                        "h-full w-[3px] flex-1 rounded-full transition duration-200 ease-out hover:-translate-y-1.5 motion-reduce:transition-none motion-reduce:hover:translate-y-0",
                        isFilled ? filledColor : "bg-zinc-300",
                      )}
                    />
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={6}>
                    {isFilled ? usedLabel : leftLabel}
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        </TooltipProvider>
        <Text variant="body" className="shrink-0 text-zinc-500">
          <span className="font-medium text-zinc-900">
            {formatFileSize(data.used_bytes)}
          </span>{" "}
          of {formatFileSize(data.limit_bytes)} used
        </Text>
      </div>
    </div>
  );
}
