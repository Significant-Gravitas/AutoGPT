"use client";

import { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import {
  formatFileSize,
  getFileDownloadUrl,
} from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { File02Icon } from "@hugeicons/core-free-icons";
import { TaskCard } from "./TaskCard";
import { useTaskFilesCard } from "./useTaskFilesCard";

interface Props {
  sessionId: string | null;
}

/** Files uploaded or generated in the session the task ran out of. The card
 *  disappears entirely (rather than showing an empty shell) when the task has
 *  no session or the session produced nothing. */
export function TaskFilesCard({ sessionId }: Props) {
  const { files } = useTaskFilesCard(sessionId);
  if (files.length === 0) return null;

  return (
    <TaskCard title="Files">
      <ul className="flex flex-col gap-2" aria-label="Task files">
        {files.map((file) => (
          <li key={file.id}>
            <TaskFile file={file} />
          </li>
        ))}
      </ul>
    </TaskCard>
  );
}

function TaskFile({ file }: { file: WorkspaceFileItem }) {
  return (
    <a
      href={getFileDownloadUrl(file.id)}
      download={file.name}
      className="flex items-center gap-2 rounded-xl bg-zinc-50 p-2.5 transition-colors hover:bg-zinc-100"
    >
      <Icon icon={File02Icon} size={16} className="shrink-0 text-zinc-400" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13px] font-medium text-zinc-900">
          {file.name}
        </span>
        <span className="block text-[11px] text-zinc-500">
          {file.origin === "uploaded" ? "Uploaded" : "Generated"} ·{" "}
          {formatFileSize(file.size_bytes)}
        </span>
      </span>
    </a>
  );
}
