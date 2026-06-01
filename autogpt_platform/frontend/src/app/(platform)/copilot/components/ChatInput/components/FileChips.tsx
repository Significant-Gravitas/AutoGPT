"use client";

import { getFileTypeIcon } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { cn } from "@/lib/utils";
import {
  CircleNotch as CircleNotchIcon,
  X as XIcon,
} from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  type Attachment,
  attachmentName,
} from "../../../helpers/workspaceAttachments";

function attachmentMimeType(attachment: Attachment): string {
  return attachment.kind === "local"
    ? attachment.file.type
    : attachment.mimeType;
}

// Stable key so AnimatePresence animates the element that actually left, not
// whatever shifted into its index.
function attachmentKey(attachment: Attachment): string {
  return attachment.kind === "workspace"
    ? `ws-${attachment.fileId}`
    : `local-${attachment.file.name}-${attachment.file.size}-${attachment.file.lastModified}`;
}

interface Props {
  attachments: Attachment[];
  onRemove: (index: number) => void;
  isUploading?: boolean;
}

// ease-out so chips settle naturally; kept under 300ms (Emil's timing rules).
const EASE_OUT = [0.16, 1, 0.3, 1] as const;
const DURATION = 0.2;

export function FileChips({ attachments, onRemove, isUploading }: Props) {
  const reduceMotion = useReducedMotion();

  return (
    <AnimatePresence initial={false}>
      {attachments.length > 0 && (
        <motion.div
          key="file-chips-row"
          initial={reduceMotion ? { opacity: 0 } : { opacity: 0, height: 0 }}
          animate={
            reduceMotion ? { opacity: 1 } : { opacity: 1, height: "auto" }
          }
          exit={reduceMotion ? { opacity: 0 } : { opacity: 0, height: 0 }}
          transition={{ duration: DURATION, ease: EASE_OUT }}
          className="w-full overflow-hidden"
        >
          <div className="flex w-full flex-wrap gap-2 px-3 pb-2 pt-2">
            <AnimatePresence initial={false} mode="popLayout">
              {attachments.map((attachment, index) => {
                const name = attachmentName(attachment);
                const Icon = getFileTypeIcon(attachmentMimeType(attachment));
                // Workspace files are already stored — only local files show
                // the upload spinner while a send is in flight.
                const showSpinner = isUploading && attachment.kind === "local";
                const restOpacity = showSpinner ? 0.7 : 1;
                return (
                  <motion.span
                    key={attachmentKey(attachment)}
                    layout
                    initial={
                      reduceMotion
                        ? { opacity: 0 }
                        : { opacity: 0, scale: 0.95, filter: "blur(4px)" }
                    }
                    animate={
                      reduceMotion
                        ? { opacity: restOpacity }
                        : {
                            opacity: restOpacity,
                            scale: 1,
                            filter: "blur(0px)",
                          }
                    }
                    exit={
                      reduceMotion
                        ? { opacity: 0 }
                        : { opacity: 0, scale: 0.95, filter: "blur(4px)" }
                    }
                    transition={{ duration: DURATION, ease: EASE_OUT }}
                    style={{ willChange: "transform, opacity, filter" }}
                    className="inline-flex items-center gap-1 rounded-full bg-zinc-100 px-3 py-1 text-sm text-zinc-700"
                  >
                    <Icon
                      weight="regular"
                      className="h-3.5 w-3.5 shrink-0 text-zinc-900"
                    />
                    <span className="max-w-[160px] truncate">{name}</span>
                    {showSpinner ? (
                      <CircleNotchIcon className="ml-0.5 h-3 w-3 animate-spin text-zinc-400" />
                    ) : (
                      <button
                        type="button"
                        aria-label={`Remove ${name}`}
                        onClick={() => onRemove(index)}
                        className="ml-0.5 rounded-full p-0.5 text-zinc-400 transition-colors hover:bg-zinc-200 hover:text-zinc-600"
                      >
                        <XIcon className="h-3 w-3" weight="bold" />
                      </button>
                    )}
                  </motion.span>
                );
              })}
            </AnimatePresence>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
