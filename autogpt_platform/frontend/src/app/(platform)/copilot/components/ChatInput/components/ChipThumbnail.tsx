"use client";

import { cn } from "@/lib/utils";
import { type ReactNode, useState } from "react";
import {
  type Attachment,
  workspaceFileDownloadUrl,
} from "../../../helpers/workspaceAttachments";
import { LocalFileImage } from "../../LocalFileImage/LocalFileImage";

interface Props {
  attachment: Attachment;
  stacked: boolean;
  /** Shown instead when the image fails to load. */
  fallback: ReactNode;
}

/** Round preview of an attached image, in place of the chip's file icon. */
export function ChipThumbnail({ attachment, stacked, fallback }: Props) {
  const [failed, setFailed] = useState(false);
  const className = cn(
    "size-[18px] shrink-0 rounded-full bg-surface-active object-cover",
    stacked ? "-ml-1" : "-ml-1.5",
  );

  if (failed) return fallback;
  if (attachment.kind === "local") {
    return (
      <LocalFileImage
        file={attachment.file}
        className={className}
        onError={() => setFailed(true)}
      />
    );
  }
  if (attachment.kind === "workspace") {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={workspaceFileDownloadUrl(attachment.fileId)}
        alt=""
        className={className}
        onError={() => setFailed(true)}
      />
    );
  }
  return null;
}
