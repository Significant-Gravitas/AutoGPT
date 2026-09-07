"use client";

import { getFileTypeIcon } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  type Attachment,
  attachmentName,
} from "../../../helpers/workspaceAttachments";
import { Cancel01Icon, Loading03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

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
  /** Card composer chips: white with a hairline ring, sitting inside the
   *  frame's own padding. */
  stacked?: boolean;
}

// ease-out so chips settle naturally; kept under 300ms (Emil's timing rules).
const EASE_OUT = [0.16, 1, 0.3, 1] as const;
const DURATION = 0.2;

export function FileChips({
  attachments,
  onRemove,
  isUploading,
  stacked = false,
}: Props) {
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
          // The pull towards the textarea belongs on the clipping box, not
          // inside it: a negative bottom margin on the row would shrink this
          // wrapper's auto height by the same amount and crop the chips.
          className={cn("w-full overflow-hidden", stacked && "-mb-1.5")}
        >
          <div
            className={cn(
              "flex w-full flex-wrap gap-2 px-3 pb-2 pt-2",
              stacked && "gap-1.5 p-0",
            )}
          >
            <AnimatePresence initial={false} mode="popLayout">
              {attachments.map((attachment, index) => {
                const name = attachmentName(attachment);
                const fileIcon = getFileTypeIcon(
                  attachmentMimeType(attachment),
                );
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
                    className={cn(
                      "inline-flex items-center gap-1 rounded-full bg-zinc-100 px-3 py-1 text-sm text-zinc-700",
                      stacked &&
                        "gap-1.5 border border-zinc-200 bg-white py-[3px] pl-2 pr-1.5 text-xs text-zinc-900 shadow-[0_1px_2px_rgba(0,0,0,0.02)]",
                    )}
                  >
                    <Icon
                      icon={fileIcon}
                      className={cn(
                        "h-3.5 w-3.5 shrink-0 text-zinc-900",
                        stacked && "text-zinc-400",
                      )}
                    />
                    <span className="max-w-[160px] truncate">{name}</span>
                    {showSpinner ? (
                      <Icon
                        icon={Loading03Icon}
                        className="ml-0.5 h-3 w-3 animate-spin text-zinc-400"
                      />
                    ) : (
                      <button
                        type="button"
                        aria-label={`Remove ${name}`}
                        onClick={() => onRemove(index)}
                        className={cn(
                          "ml-0.5 rounded-full p-0.5 text-zinc-400 transition-colors hover:bg-zinc-200 hover:text-zinc-600",
                          stacked && "hover:bg-zinc-100 hover:text-zinc-900",
                        )}
                      >
                        <Icon icon={Cancel01Icon} className="h-3 w-3" />
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
