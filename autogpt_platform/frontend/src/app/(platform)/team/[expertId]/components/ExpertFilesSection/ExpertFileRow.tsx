"use client";

import { Download01Icon } from "@hugeicons/core-free-icons";
import { FileThumbnail } from "@/app/(platform)/artifacts/components/ArtifactsList/ArtifactsTable/FileThumbnail";
import {
  formatFileSize,
  formatRelativeDate,
  getFileDownloadUrl,
} from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  file: WorkspaceFileItem;
}

/** A Files-page style row: thumbnail, name, when and how big. The whole
 *  row opens the file in a new tab. */
export function ExpertFileRow({ file }: Props) {
  return (
    <a
      href={getFileDownloadUrl(file.id)}
      target="_blank"
      rel="noreferrer"
      title={file.name}
      className="group flex items-center gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-surface-muted"
    >
      <FileThumbnail file={file} />
      <span className="flex min-w-0 flex-1 flex-col">
        <Text
          variant="body-md-medium"
          as="span"
          className="truncate !text-foreground-strong"
        >
          {file.name}
        </Text>
        <Text variant="small" as="span" className="!text-foreground-tertiary">
          {formatRelativeDate(file.created_at)} ·{" "}
          {formatFileSize(file.size_bytes)}
        </Text>
      </span>
      <Icon
        icon={Download01Icon}
        size={16}
        aria-hidden
        className="shrink-0 text-foreground-subtle opacity-0 transition-opacity group-hover:opacity-100 [@media(hover:none)]:opacity-100"
      />
    </a>
  );
}
