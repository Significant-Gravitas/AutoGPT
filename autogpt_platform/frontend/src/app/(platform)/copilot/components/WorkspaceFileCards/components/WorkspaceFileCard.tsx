"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { classifyArtifact } from "../../ArtifactPanel/helpers";
import type { SessionFile } from "../../ContextPanel/components/FilesTab/useSessionFiles";
import { FileRowActions } from "./FileRowActions";

interface Props {
  file: SessionFile;
  onOpen: (file: SessionFile) => void;
  onDownload: (file: SessionFile) => void;
  onRequestDelete: (file: SessionFile) => void;
}

export function WorkspaceFileCard({
  file,
  onOpen,
  onDownload,
  onRequestDelete,
}: Props) {
  const { item } = file;
  const fileIcon = classifyArtifact(item.mime_type ?? null, item.name).icon;

  return (
    <div className="group relative -mx-2.5 flex items-center gap-3 rounded-xl px-2.5 py-1.5 transition-colors hover:bg-zinc-50">
      <button
        type="button"
        onClick={() => onOpen(file)}
        title={item.name}
        className="flex min-w-0 flex-1 items-center gap-3 text-left"
      >
        <Icon icon={fileIcon} size={18} className="shrink-0 text-zinc-700" />
        {/* Long names fade out rather than ellipsing, so the row keeps a clean
            edge next to the hover actions. */}
        <span className="min-w-0 flex-1 overflow-hidden whitespace-nowrap text-[15px] text-zinc-800 [mask-image:linear-gradient(to_right,black_calc(100%_-_2rem),transparent)]">
          {item.name}
        </span>
      </button>
      <FileRowActions
        file={file}
        onDownload={onDownload}
        onRequestDelete={onRequestDelete}
      />
    </div>
  );
}
