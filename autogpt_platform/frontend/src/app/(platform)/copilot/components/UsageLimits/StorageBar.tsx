"use client";

import { Text } from "@/components/atoms/Text/Text";
import { formatBytes } from "../usageHelpers";
import { useWorkspaceStorage } from "./useWorkspaceStorage";

export function StorageBar() {
  const { data: storage } = useWorkspaceStorage();
  if (!storage || storage.limit_bytes <= 0) return null;

  const { used_bytes, limit_bytes, file_count } = storage;
  const percent = Math.min(100, Math.round((used_bytes / limit_bytes) * 100));
  const isHigh = percent >= 80;
  const percentLabel =
    used_bytes > 0 && percent === 0 ? "<1% used" : `${percent}% used`;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <Text variant="body-medium" className="text-neutral-700">
          File storage
        </Text>
        <Text variant="body" className="tabular-nums text-neutral-500">
          {percentLabel}
        </Text>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200">
        <div
          role="progressbar"
          aria-label="File storage usage"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          className={`h-full rounded-full transition-[width] duration-300 ease-out ${
            isHigh ? "bg-orange-500" : "bg-blue-500"
          }`}
          style={{ width: `${Math.max(used_bytes > 0 ? 1 : 0, percent)}%` }}
        />
      </div>
      <Text variant="small" className="text-neutral-400">
        {formatBytes(used_bytes)} of {formatBytes(limit_bytes)} &middot;{" "}
        {file_count} {file_count === 1 ? "file" : "files"}
      </Text>
    </div>
  );
}
