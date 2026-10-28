"use client";

import { getFileTypeIcon } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { isPreviewableImage } from "../../ArtifactPanel/helpers";
import { ChipThumbnail } from "./ChipThumbnail";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  type Attachment,
  attachmentKey,
  attachmentName,
} from "../../../helpers/workspaceAttachments";
import { folderSummary } from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
import {
  Cancel01Icon,
  Folder01Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

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
                const isFolder = attachment.kind === "folder";
                const mimeType =
                  attachment.kind === "local"
                    ? attachment.file.type
                    : attachment.kind === "workspace"
                      ? attachment.mimeType
                      : "";
                const chipIcon = (
                  <Icon
                    icon={isFolder ? Folder01Icon : getFileTypeIcon(mimeType)}
                    className={cn(
                      "h-3.5 w-3.5 shrink-0 text-foreground-strong",
                      stacked && "text-foreground-subtle",
                    )}
                  />
                );
                const count = isFolder
                  ? folderSummary(
                      attachment.fileCount,
                      attachment.subfolderCount,
                    )
                  : null;
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
                    aria-label={
                      isFolder ? `Folder: ${name}, ${count}` : undefined
                    }
                    className={cn(
                      "inline-flex items-center gap-1 rounded-full bg-surface-hover px-3 py-1 text-sm text-foreground-body",
                      stacked &&
                        // TODO(tokens): no exact role for shadow-[0_1px_2px_rgba(0,0,0,0.02)]
                        "gap-1.5 border border-border-base bg-surface py-[3px] pl-2 pr-1.5 text-xs text-foreground-strong shadow-[0_1px_2px_rgba(0,0,0,0.02)]",
                    )}
                  >
                    {isPreviewableImage(mimeType) ? (
                      <ChipThumbnail
                        attachment={attachment}
                        stacked={stacked}
                        fallback={chipIcon}
                      />
                    ) : (
                      chipIcon
                    )}
                    <span className="max-w-[160px] truncate">{name}</span>
                    {count ? (
                      <span className="shrink-0 text-foreground-tertiary">
                        · {count}
                      </span>
                    ) : null}
                    {showSpinner ? (
                      <Icon
                        icon={Loading03Icon}
                        className="ml-0.5 h-3 w-3 animate-spin text-foreground-subtle"
                      />
                    ) : (
                      <button
                        type="button"
                        aria-label={
                          isFolder ? `Remove folder ${name}` : `Remove ${name}`
                        }
                        onClick={() => onRemove(index)}
                        className={cn(
                          "ml-0.5 rounded-full p-0.5 text-foreground-subtle transition-colors hover:bg-surface-active hover:text-foreground-secondary",
                          stacked &&
                            "hover:bg-surface-hover hover:text-foreground-strong",
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
