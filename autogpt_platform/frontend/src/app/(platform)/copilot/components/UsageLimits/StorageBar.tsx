import { cn } from "@/lib/utils";
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
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-neutral-700">
          File storage
        </span>
        <span className="text-[11px] tabular-nums text-neutral-500">
          {percentLabel}
        </span>
      </div>
      <div className="text-[10px] text-neutral-400">
        {formatBytes(used_bytes)} of {formatBytes(limit_bytes)} &middot;{" "}
        {file_count} {file_count === 1 ? "file" : "files"}
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200">
        <div
          className={cn(
            "h-full rounded-full transition-[width] duration-300 ease-out",
            isHigh ? "bg-orange-500" : "bg-blue-500",
          )}
          style={{ width: `${Math.max(used_bytes > 0 ? 1 : 0, percent)}%` }}
        />
      </div>
    </div>
  );
}
