"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { PreviewBody } from "../ArtifactCard/CardPreview";
import {
  formatDayLabel,
  formatFileSize,
  getFileTypeIcon,
  getFileTypeLabel,
  getPreviewKind,
} from "../helpers";

interface Props {
  file: WorkspaceFileItem;
}

// Wide enough that the backend thumbnail stays sharp on 2x displays.
export const PREVIEW_IMAGE_WIDTH = 800;

export function FilePreviewCard({ file }: Props) {
  const kind = getPreviewKind(file.mime_type, file.size_bytes, file.name);

  return (
    <div className="w-80 overflow-hidden" data-testid="artifacts-preview-card">
      <div className="relative aspect-[16/10] overflow-hidden border-b border-border-base bg-surface-muted">
        <PreviewBody file={file} kind={kind} imageWidth={PREVIEW_IMAGE_WIDTH} />
      </div>
      <div className="flex items-center gap-3 p-3">
        <Icon
          icon={getFileTypeIcon(file.mime_type, file.name)}
          size={20}
          className="shrink-0 text-foreground-tertiary"
        />
        <div className="flex min-w-0 flex-1 flex-col">
          <Text
            variant="body-medium"
            as="span"
            className="truncate text-foreground-strong"
          >
            {file.name}
          </Text>
          <Text
            variant="small"
            as="span"
            className="truncate text-foreground-tertiary"
          >
            {getFileTypeLabel(file.mime_type, file.name)} ·{" "}
            {formatFileSize(file.size_bytes)} ·{" "}
            {formatDayLabel(file.created_at)}
          </Text>
        </div>
      </div>
    </div>
  );
}
