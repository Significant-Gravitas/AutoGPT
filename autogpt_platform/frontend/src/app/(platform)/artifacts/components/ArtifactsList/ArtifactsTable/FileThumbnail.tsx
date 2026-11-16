"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useState } from "react";
import {
  getFilePreviewUrl,
  getFileTypeIcon,
  getPreviewKind,
  hasImageThumbnail,
} from "../helpers";

interface Props {
  file: WorkspaceFileItem;
}

export const THUMBNAIL_WIDTH = 96;

export function FileThumbnail({ file }: Props) {
  const [hasError, setHasError] = useState(false);
  const kind = getPreviewKind(file.mime_type, file.size_bytes, file.name);
  const showImage = hasImageThumbnail(kind) && !hasError;

  return (
    <div
      className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-xl border border-border-base bg-surface"
      data-testid="artifacts-thumbnail"
    >
      {showImage ? (
        // eslint-disable-next-line @next/next/no-img-element -- the preview endpoint already sizes the thumbnail
        <img
          src={getFilePreviewUrl(file.id, { width: THUMBNAIL_WIDTH })}
          alt=""
          loading="lazy"
          onError={() => setHasError(true)}
          className="h-full w-full object-cover"
        />
      ) : (
        <span className="flex text-foreground-tertiary">
          <Icon icon={getFileTypeIcon(file.mime_type, file.name)} size={18} />
        </span>
      )}
    </div>
  );
}
