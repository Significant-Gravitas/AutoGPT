"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { getFileDownloadUrl, getFilePreviewUrl } from "../helpers";
import { LoadingPlaceholder } from "./PreviewParts";

interface PreviewProps {
  file: WorkspaceFileItem;
  onError: () => void;
}

// Shared by image / pdf / office kinds — the backend returns a small WebP
// thumbnail for all three, so the client just paints an <img>.
export function ImagePreview({ file, onError }: PreviewProps) {
  const [isLoaded, setIsLoaded] = useState(false);

  return (
    <>
      {!isLoaded ? <LoadingPlaceholder file={file} /> : null}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={getFilePreviewUrl(file.id, { width: 400 })}
        alt={file.name}
        loading="lazy"
        onLoad={() => setIsLoaded(true)}
        onError={onError}
        className={cn(
          "h-full w-full object-cover transition-opacity duration-300",
          isLoaded ? "opacity-100" : "opacity-0",
        )}
      />
    </>
  );
}

export function VideoPreview({ file, onError }: PreviewProps) {
  const [isLoaded, setIsLoaded] = useState(false);

  return (
    <>
      {!isLoaded ? <LoadingPlaceholder file={file} /> : null}
      <video
        src={getFileDownloadUrl(file.id)}
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
    </>
  );
}
