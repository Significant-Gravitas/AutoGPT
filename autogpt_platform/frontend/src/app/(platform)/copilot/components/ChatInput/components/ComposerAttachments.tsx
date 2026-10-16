"use client";

import { folderSummary } from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef } from "react";
import {
  type Attachment,
  attachmentKey,
  attachmentName,
  workspaceFileDownloadUrl,
} from "../../../helpers/workspaceAttachments";
import { AttachmentCard } from "../../AttachmentCard/AttachmentCard";
import { FOLDER_MEDIA_TYPE, isImageType } from "../../AttachmentCard/helpers";
import { useObjectUrl } from "../../AttachmentCard/useObjectUrl";

interface Props {
  attachments: Attachment[];
  onRemove: (index: number) => void;
  /** Card composer: the row sits inside the frame's own padding. */
  stacked?: boolean;
}

const EASE_OUT = [0.16, 1, 0.3, 1] as const;
const DURATION = 0.2;

/** The attached files as cards above the composer's text line. */
export function ComposerAttachments({
  attachments,
  onRemove,
  stacked = false,
}: Props) {
  const reduceMotion = useReducedMotion();
  const listRef = useRef<HTMLUListElement>(null);
  const count = attachments.length;
  const previousCount = useRef(count);

  useEffect(() => {
    const list = listRef.current;
    if (list && count > previousCount.current)
      list.scrollTop = list.scrollHeight;
    previousCount.current = count;
  }, [count]);

  const hidden = reduceMotion
    ? { opacity: 0 }
    : { opacity: 0, scale: 0.96, filter: "blur(4px)" };
  const shown = reduceMotion
    ? { opacity: 1 }
    : { opacity: 1, scale: 1, filter: "blur(0px)" };

  return (
    <AnimatePresence initial={false}>
      {attachments.length > 0 && (
        <motion.div
          key="composer-attachments"
          initial={reduceMotion ? { opacity: 0 } : { opacity: 0, height: 0 }}
          animate={
            reduceMotion ? { opacity: 1 } : { opacity: 1, height: "auto" }
          }
          exit={reduceMotion ? { opacity: 0 } : { opacity: 0, height: 0 }}
          transition={{ duration: DURATION, ease: EASE_OUT }}
          className="w-full overflow-hidden"
        >
          <ul
            ref={listRef}
            aria-label="Attachments"
            className={cn(
              "flex max-h-[min(40vh,14rem)] w-full flex-wrap items-center gap-2 overflow-y-auto px-3 pb-1 pt-3 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300",
              stacked && "p-0",
            )}
          >
            <AnimatePresence initial={false} mode="popLayout">
              {attachments.map((attachment, index) => (
                <motion.li
                  key={attachmentKey(attachment)}
                  layout={!reduceMotion}
                  initial={hidden}
                  animate={shown}
                  exit={hidden}
                  transition={{ duration: DURATION, ease: EASE_OUT }}
                  className="min-w-0 max-w-full"
                >
                  <ComposerAttachment
                    attachment={attachment}
                    onRemove={() => onRemove(index)}
                  />
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

interface AttachmentProps {
  attachment: Attachment;
  onRemove: () => void;
}

function ComposerAttachment({ attachment, onRemove }: AttachmentProps) {
  const name = attachmentName(attachment);
  const localImage =
    attachment.kind === "local" && isImageType(attachment.file.type)
      ? attachment.file
      : undefined;
  const previewUrl = useObjectUrl(localImage);

  if (attachment.kind === "folder")
    return (
      <AttachmentCard
        size="composer"
        name={name}
        mediaType={FOLDER_MEDIA_TYPE}
        detail={`Folder · ${folderSummary(attachment.fileCount, attachment.subfolderCount)}`}
        onRemove={onRemove}
      />
    );

  if (attachment.kind === "local")
    return (
      <AttachmentCard
        size="composer"
        name={name}
        mediaType={attachment.file.type || "application/octet-stream"}
        sizeBytes={attachment.file.size}
        previewUrl={previewUrl}
        onRemove={onRemove}
      />
    );

  return (
    <AttachmentCard
      size="composer"
      name={name}
      mediaType={attachment.mimeType}
      sizeBytes={attachment.sizeBytes}
      previewUrl={workspaceFileDownloadUrl(attachment.fileId)}
      onRemove={onRemove}
    />
  );
}
