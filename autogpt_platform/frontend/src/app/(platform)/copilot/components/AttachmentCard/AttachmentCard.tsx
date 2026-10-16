"use client";

import { useState } from "react";
import type { PendingUploadStatus } from "../../copilotStreamStore";
import { FileAttachment } from "./components/FileAttachment";
import { ImageAttachment } from "./components/ImageAttachment";
import { isImageType } from "./helpers";

export interface AttachmentCardProps {
  name: string;
  /** The file's MIME type, or `FOLDER_MEDIA_TYPE` for a workspace folder. */
  mediaType: string;
  sizeBytes?: number;
  /** Replaces the "PDF · 1.2 MB" line, e.g. a folder's file count. */
  detail?: string;
  /** Thumbnail source for an image; a placeholder shows until it is set. */
  previewUrl?: string;
  /** Upload state, for a message whose files are still on their way. */
  status?: PendingUploadStatus;
  size: "composer" | "message";
  onRemove?: () => void;
  onRetry?: () => void;
  onOpen?: () => void;
  href?: string;
}

/**
 * One attachment, as ChatGPT draws them: an image as a square thumbnail,
 * anything else as a card with its file-type icon, name, and type · size.
 * The same card sits above the composer's text line and above a sent
 * message, so a file looks the same before and after it is sent.
 */
export function AttachmentCard(props: AttachmentCardProps) {
  const [hasPreviewFailed, setHasPreviewFailed] = useState(false);
  // A failed upload needs room for its reason and its Retry, so it always
  // takes the card layout; so does an image the browser cannot draw.
  const showsThumbnail =
    isImageType(props.mediaType) &&
    props.status !== "failed" &&
    !hasPreviewFailed;

  return (
    <div
      data-testid="attachment-card"
      data-status={props.status}
      className="relative min-w-0 max-w-full"
    >
      {showsThumbnail ? (
        <ImageAttachment
          {...props}
          onPreviewError={() => setHasPreviewFailed(true)}
        />
      ) : (
        <FileAttachment {...props} />
      )}
    </div>
  );
}
