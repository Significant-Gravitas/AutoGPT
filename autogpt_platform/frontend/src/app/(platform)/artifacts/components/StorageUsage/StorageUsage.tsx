"use client";

import { useGetWorkspaceStorageUsage } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { Progress } from "@/components/atoms/Progress/Progress";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatFileSize } from "../ArtifactsList/helpers";

export function StorageUsage() {
  const { data, isLoading, isError } = useGetWorkspaceStorageUsage({
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
    },
  });

  if (isLoading) {
    return (
      <div
        className="flex items-center gap-3"
        data-testid="storage-usage-loading"
      >
        <Skeleton className="h-1.5 w-32 rounded-full" />
        <Skeleton className="h-3 w-28" />
      </div>
    );
  }

  if (isError || !data) return null;

  const percent = Math.min(Math.max(data.used_percent, 0), 100);
  const usedLabel = `${Math.round(percent)}% of storage used`;

  return (
    <div
      className="flex items-center gap-3"
      title={usedLabel}
      data-testid="storage-usage"
    >
      <Text variant="small" as="span" className="text-zinc-500">
        Storage
      </Text>
      <Progress
        value={percent}
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={usedLabel}
        className={cn(
          "h-1.5 w-32 bg-zinc-100",
          percent > 95 ? "[&>div]:bg-red-400" : "[&>div]:bg-zinc-400",
        )}
      />
      <Text variant="small" as="span" className="text-zinc-500">
        <span className="font-medium text-zinc-700">
          {formatFileSize(data.used_bytes)}
        </span>{" "}
        of {formatFileSize(data.limit_bytes)}
      </Text>
    </div>
  );
}
