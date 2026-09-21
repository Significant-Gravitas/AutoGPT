"use client";

import { getFilePreviewUrl } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { useEffect, useState } from "react";
import type { Attachment } from "../../../helpers/workspaceAttachments";

interface Props {
  attachment: Attachment;
  name: string;
  onError: () => void;
}

const THUMBNAIL_WIDTH = 128;

function useAttachmentPreviewUrl(attachment: Attachment): string | null {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const localFile = attachment.kind === "local" ? attachment.file : null;

  useEffect(() => {
    if (!localFile) return;
    const url = URL.createObjectURL(localFile);
    setObjectUrl(url);
    return () => {
      URL.revokeObjectURL(url);
      setObjectUrl(null);
    };
  }, [localFile]);

  if (attachment.kind === "workspace") {
    return getFilePreviewUrl(attachment.fileId, { width: THUMBNAIL_WIDTH });
  }
  return objectUrl;
}

export function AttachmentThumbnail({ attachment, name, onError }: Props) {
  const src = useAttachmentPreviewUrl(attachment);
  if (!src) return null;
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={name}
      data-testid="attachment-thumbnail"
      onError={onError}
      className="h-full w-full object-cover"
    />
  );
}
