"use client";

import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import { FileThumbnail } from "@/app/(platform)/artifacts/components/ArtifactsList/ArtifactsTable/FileThumbnail";
import {
  formatDayLabel,
  formatFileSize,
  formatFullDate,
} from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { Text } from "@/components/atoms/Text/Text";
import type { SessionFile } from "../../ContextPanel/components/FilesTab/useSessionFiles";
import { FileRowActions } from "./FileRowActions";

interface Props {
  file: SessionFile;
  onOpen: (file: SessionFile) => void;
  onDownload: (file: SessionFile) => void;
  onRequestDelete: (file: SessionFile) => void;
}

/** A Files-page style row (thumbnail, name, date and size) sized for the
 *  side panel. */
export function WorkspaceFileRow({
  file,
  onOpen,
  onDownload,
  onRequestDelete,
}: Props) {
  const { item } = file;
  // Relative to the clock, so read on every render rather than cached.
  const dayLabel = useReadEveryRender(() => formatDayLabel(item.created_at));

  return (
    <div className="group flex items-center gap-3 px-3 py-2 transition-colors hover:bg-surface-hover/70">
      <button
        type="button"
        onClick={() => onOpen(file)}
        title={item.name}
        className="flex min-w-0 flex-1 items-center gap-3 text-left"
      >
        <FileThumbnail file={item} />
        <span className="flex min-w-0 flex-col">
          <Text
            variant="body-medium"
            as="span"
            className="truncate text-foreground-strong"
          >
            {item.name}
          </Text>
          <Text
            variant="small"
            as="span"
            className="text-foreground-tertiary"
            title={formatFullDate(item.created_at)}
          >
            {dayLabel} · {formatFileSize(item.size_bytes)}
          </Text>
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
