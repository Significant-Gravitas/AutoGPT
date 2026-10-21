"use client";

import { cn } from "@/lib/utils";
import {
  type Attachment,
  workspaceFileDownloadUrl,
} from "../../../helpers/workspaceAttachments";
import { LocalFileImage } from "../../LocalFileImage/LocalFileImage";

interface Props {
  attachment: Attachment;
  stacked: boolean;
}

/** Round preview of an attached image, in place of the chip's file icon. */
export function ChipThumbnail({ attachment, stacked }: Props) {
  const className = cn(
    "size-[18px] shrink-0 rounded-full bg-zinc-200 object-cover",
    stacked ? "-ml-1" : "-ml-1.5",
  );

  if (attachment.kind === "local") {
    return <LocalFileImage file={attachment.file} className={className} />;
  }
  if (attachment.kind === "workspace") {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={workspaceFileDownloadUrl(attachment.fileId)}
        alt=""
        className={className}
      />
    );
  }
  return null;
}
