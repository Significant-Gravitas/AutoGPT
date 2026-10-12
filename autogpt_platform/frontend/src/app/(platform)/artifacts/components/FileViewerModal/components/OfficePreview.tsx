"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { PREVIEW_IMAGE_WIDTH } from "../../ArtifactsList/ArtifactsTable/FilePreviewCard";
import {
  getFilePreviewUrl,
  getFileTypeLabel,
} from "../../ArtifactsList/helpers";
import { DownloadOnly } from "./DownloadOnly";

interface Props {
  file: WorkspaceFileItem;
  downloadUrl: string;
}

// The preview endpoint returns the thumbnail embedded in the office file
// (first slide or page), and 415s when there is none.
export function OfficePreview({ file, downloadUrl }: Props) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  if (hasError) {
    return <DownloadOnly name={file.name} downloadUrl={downloadUrl} />;
  }

  const isPresentation =
    getFileTypeLabel(file.mime_type, file.name) === "Presentation";

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="relative flex min-h-0 flex-1 items-center justify-center">
        {isLoaded ? null : <Skeleton className="absolute inset-0" />}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={getFilePreviewUrl(file.id, { width: PREVIEW_IMAGE_WIDTH })}
          alt={file.name}
          className={cn(
            "max-h-full max-w-full object-contain",
            isLoaded ? "opacity-100" : "opacity-0",
          )}
          onLoad={() => setIsLoaded(true)}
          onError={() => setHasError(true)}
          data-testid="file-viewer-office-preview"
        />
      </div>
      <div className="shrink-0">
        <DownloadOnly
          name={file.name}
          downloadUrl={downloadUrl}
          message={
            isPresentation
              ? "Showing the first slide. Download to open the full presentation."
              : "Showing the first page. Download to open the full file."
          }
        />
      </div>
    </div>
  );
}
