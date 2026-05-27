import { DownloadSimpleIcon, TrashIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { classifyArtifact } from "../../../../ArtifactPanel/helpers";
import { formatFileSize, formatFileTimestamp, isUploadedFile } from "../helpers";
import type { SessionFile } from "../useSessionFiles";

interface Props {
  file: SessionFile;
  onOpen: (file: SessionFile) => void;
  onDownload: (file: SessionFile) => void;
  onRequestDelete: (file: SessionFile) => void;
}

export function FileRow({ file, onOpen, onDownload, onRequestDelete }: Props) {
  const { item } = file;
  const Icon = classifyArtifact(item.mime_type ?? null, item.name).icon;
  const canDelete = !isUploadedFile(item);

  return (
    <div className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-zinc-50">
      <button
        type="button"
        onClick={() => onOpen(file)}
        className="flex min-w-0 flex-1 items-center gap-2 text-left"
        title={file.messageID ? "Jump to chat" : item.name}
      >
        <Icon size={18} className="shrink-0 text-zinc-400" />
        <span className="flex min-w-0 flex-col">
          <span className="truncate text-sm text-zinc-800">{item.name}</span>
          <span className="text-xs text-zinc-400">
            {formatFileSize(item.size_bytes ?? 0)} · {formatFileTimestamp(item.created_at)}
          </span>
        </span>
      </button>
      <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
        <button
          type="button"
          onClick={() => onDownload(file)}
          aria-label={`Download ${item.name}`}
          className="rounded p-1.5 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700"
        >
          <DownloadSimpleIcon size={16} />
        </button>
        {canDelete && (
          <button
            type="button"
            onClick={() => onRequestDelete(file)}
            aria-label={`Delete ${item.name}`}
            className={cn("rounded p-1.5 text-zinc-500 hover:bg-red-50 hover:text-red-600")}
          >
            <TrashIcon size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
