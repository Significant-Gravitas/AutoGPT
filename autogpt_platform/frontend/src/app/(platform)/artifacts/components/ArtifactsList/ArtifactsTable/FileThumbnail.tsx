"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useFileBlobUrl } from "../ArtifactCard/PreviewParts";
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
  const previewUrl = useFileBlobUrl(
    file,
    hasImageThumbnail(kind)
      ? getFilePreviewUrl(file.id, { width: THUMBNAIL_WIDTH })
      : null,
    () => setHasError(true),
  );
  const showImage = hasImageThumbnail(kind) && !hasError;

  return (
    <div
      className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-xl border border-zinc-200 bg-white"
      data-testid="artifacts-thumbnail"
    >
      {showImage && previewUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={previewUrl}
          alt=""
          loading="lazy"
          onError={() => setHasError(true)}
          className="h-full w-full object-cover"
        />
      ) : (
        <Icon
          icon={getFileTypeIcon(file.mime_type, file.name)}
          size={18}
          className="text-zinc-500"
        />
      )}
    </div>
  );
}
