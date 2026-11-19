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
  const internalBytes = data.internal_bytes ?? 0;

  return (
    <div
      className="flex items-center gap-3"
      title={usedLabel}
      data-testid="storage-usage"
    >
      <Text variant="small" as="span" tone="secondary">
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
          // oxlint-disable-next-line design-system/no-restyle -- a slim grey meter; Progress has no size or tone variants
          "h-1.5 w-32 bg-surface-hover",
          // oxlint-disable-next-line design-system/no-restyle -- the meter turns red near the quota
          percent > 95 ? "[&>div]:bg-error" : "[&>div]:bg-foreground-subtle",
        )}
      />
      <Text variant="small" as="span" tone="secondary">
        <span className="font-medium text-foreground-body">
          {formatFileSize(data.used_bytes)}
        </span>{" "}
        of {formatFileSize(data.limit_bytes)}
      </Text>
      {internalBytes > 0 ? (
        <Text
          variant="small"
          as="span"
          tone="secondary"
          className="hidden lg:inline"
        >
          {`${formatFileSize(internalBytes)} is your team's skills and working files`}
        </Text>
      ) : null}
    </div>
  );
}
