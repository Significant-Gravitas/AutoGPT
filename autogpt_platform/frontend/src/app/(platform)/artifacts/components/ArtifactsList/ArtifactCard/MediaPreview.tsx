"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { getFileDownloadUrl, getFilePreviewUrl } from "../helpers";
import { LoadingPlaceholder, useFileBlobUrl } from "./PreviewParts";

interface PreviewProps {
  file: WorkspaceFileItem;
  onError: () => void;
}

// Shared by image / pdf / office kinds — the backend returns a small WebP
// thumbnail for all three, so the client just paints an <img>.
export function ImagePreview({ file, onError }: PreviewProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const previewUrl = useFileBlobUrl(
    file,
    getFilePreviewUrl(file.id, { width: 400 }),
    onError,
  );

  return (
    <>
      {!isLoaded ? <LoadingPlaceholder file={file} /> : null}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      {previewUrl ? (
        <img
          src={previewUrl}
          alt={file.name}
          loading="lazy"
          onLoad={() => setIsLoaded(true)}
          onError={onError}
          className={cn(
            "h-full w-full object-cover transition-opacity duration-300",
            isLoaded ? "opacity-100" : "opacity-0",
          )}
        />
      ) : null}
    </>
  );
}

export function VideoPreview({ file, onError }: PreviewProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const previewUrl = useFileBlobUrl(file, getFileDownloadUrl(file.id), onError);

  return (
    <>
      {!isLoaded ? <LoadingPlaceholder file={file} /> : null}
      {previewUrl ? (
        <video
          src={previewUrl}
          preload="metadata"
          muted
          playsInline
          // Reveal on the first painted frame when available (avoids a blank
          // box); fall back to metadata so the video is never left hidden when
          // the browser only fetches metadata under preload="metadata".
          onLoadedData={() => setIsLoaded(true)}
          onLoadedMetadata={() => setIsLoaded(true)}
          onError={onError}
          className={cn(
            "h-full w-full object-cover transition-opacity duration-300",
            isLoaded ? "opacity-100" : "opacity-0",
          )}
        />
      ) : null}
    </>
  );
}
