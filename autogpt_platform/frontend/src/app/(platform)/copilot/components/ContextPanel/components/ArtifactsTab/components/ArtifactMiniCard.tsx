"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Download04Icon } from "@hugeicons/core-free-icons";
import { classifyArtifact } from "../../../../ArtifactPanel/helpers";
import { formatFileSize, formatFileTimestamp } from "../../FilesTab/helpers";
import type { SessionFile } from "../../FilesTab/useSessionFiles";

interface Props {
  file: SessionFile;
  onOpen: (file: SessionFile) => void;
  onDownload: (file: SessionFile) => void;
}

export function ArtifactMiniCard({ file, onOpen, onDownload }: Props) {
  const { item } = file;
  const fileIcon = classifyArtifact(item.mime_type ?? null, item.name).icon;

  return (
    <div className="group relative flex items-center gap-2.5 rounded-2xl bg-white px-3.5 py-2.5 transition-transform duration-150 ease-out smooth-shadow-ring-sm hover:-translate-y-px motion-reduce:transition-none">
      <button
        type="button"
        onClick={() => onOpen(file)}
        className="flex min-w-0 flex-1 items-center gap-2.5 text-left"
        title={item.name}
      >
        <Icon icon={fileIcon} size={18} className="shrink-0 text-zinc-500" />
        <span className="flex min-w-0 flex-col">
          <span className="truncate text-sm font-medium text-zinc-800">
            {item.name}
          </span>
          <span className="truncate text-xs text-zinc-400">
            {formatFileSize(item.size_bytes ?? 0)} ·{" "}
            {formatFileTimestamp(item.created_at)}
          </span>
        </span>
      </button>
      <button
        type="button"
        onClick={() => onDownload(file)}
        aria-label={`Download ${item.name}`}
        className="shrink-0 rounded-full p-1 text-zinc-300 opacity-0 transition-opacity duration-150 ease-out group-hover:opacity-100 hover:text-zinc-600 focus-visible:opacity-100 motion-reduce:transition-none [@media(hover:none)]:opacity-100"
      >
        <Icon icon={Download04Icon} size={16} />
      </button>
    </div>
  );
}
