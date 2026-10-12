import { File02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { DelegationFile } from "../../../../../delegations";

interface Props {
  response: string | null;
  files: DelegationFile[];
}

function formatBytes(bytes: number | null): string | null {
  if (bytes === null) return null;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** What the teammate sent back: their last words and the files. */
export function DetailOutcome({ response, files }: Props) {
  return (
    <div className="flex flex-col gap-2">
      {response && (
        <p className="line-clamp-6 whitespace-pre-wrap text-sm text-zinc-700">
          {response}
        </p>
      )}
      {files.map((file) => (
        <span key={file.path} className="flex items-start gap-2 text-sm">
          <Icon
            icon={File02Icon}
            size={16}
            className="mt-0.5 shrink-0 text-zinc-500"
          />
          <span className="flex min-w-0 flex-col">
            <span className="truncate text-zinc-900">{file.name}</span>
            {formatBytes(file.sizeBytes) && (
              <span className="text-xs text-zinc-500">
                {formatBytes(file.sizeBytes)}
              </span>
            )}
          </span>
        </span>
      ))}
    </div>
  );
}
