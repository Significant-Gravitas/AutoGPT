"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ArrowLeftIcon } from "@/components/atoms/AGPTIcon/icons";
import { STATUS_LABELS } from "../../helpers";
import { DiffView } from "./DiffView";
import { useChangesTab } from "./useChangesTab";

interface Props {
  sessionId: string;
}

export function ChangesTab({ sessionId }: Props) {
  const {
    changes,
    isLoading,
    selectedPath,
    setSelectedPath,
    diff,
    isDiffLoading,
  } = useChangesTab(sessionId);

  if (selectedPath) {
    return (
      <div className="flex h-full min-h-0 flex-col">
        <button
          type="button"
          onClick={() => setSelectedPath(null)}
          className="flex items-center gap-1.5 px-3 py-2 text-sm text-zinc-600 hover:text-zinc-900"
        >
          <ArrowLeftIcon size={14} />
          <span className="truncate font-mono text-xs">{selectedPath}</span>
        </button>
        <div className="min-h-0 flex-1 border-t border-t-zinc-100">
          {isDiffLoading || !diff ? (
            <div className="p-3">
              <Skeleton className="h-full w-full" />
            </div>
          ) : (
            <DiffView
              path={selectedPath}
              original={diff.original}
              modified={diff.modified}
            />
          )}
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2 p-3">
        <Skeleton className="h-6 w-full" />
        <Skeleton className="h-6 w-4/5" />
        <Skeleton className="h-6 w-3/5" />
      </div>
    );
  }

  const files = changes?.files ?? [];
  if (!changes?.is_git_repo || files.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-zinc-400">
        No changes yet
      </div>
    );
  }

  return (
    <ul className="h-full overflow-auto py-1">
      {files.map((file) => (
        <li key={file.path}>
          <button
            type="button"
            onClick={() => setSelectedPath(file.path)}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-zinc-700 hover:bg-zinc-100"
          >
            <span className="w-4 shrink-0 text-center font-mono text-xs font-semibold text-zinc-500">
              {STATUS_LABELS[file.status] ?? file.status}
            </span>
            <span className="truncate font-mono text-xs">{file.path}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
