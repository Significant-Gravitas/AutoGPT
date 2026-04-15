"use client";

import { Button } from "@/components/atoms/Button/Button";
import { ClockCounterClockwiseIcon, XIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/utils/time";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useDraftRecoveryPopup } from "./useDraftRecoveryPopup";
import { Text } from "@/components/atoms/Text/Text";
import { AnimatePresence, motion } from "framer-motion";
import { DraftDiff } from "@/lib/dexie/draft-utils";

interface DraftRecoveryPopupProps {
  isInitialLoadComplete: boolean;
}

function formatDiffSummary(diff: DraftDiff | null): string {
  if (!diff) return "";

  const parts: string[] = [];

  // Node changes
  const nodeChanges: string[] = [];
  if (diff.nodes.added > 0) nodeChanges.push(`+${diff.nodes.added}`);
  if (diff.nodes.removed > 0) nodeChanges.push(`-${diff.nodes.removed}`);
  if (diff.nodes.modified > 0) nodeChanges.push(`~${diff.nodes.modified}`);

  if (nodeChanges.length > 0) {
    parts.push(
      `${nodeChanges.join("/")} block${diff.nodes.added + diff.nodes.removed + diff.nodes.modified !== 1 ? "s" : ""}`,
    );
  }

  // Edge changes
  const edgeChanges: string[] = [];
  if (diff.edges.added > 0) edgeChanges.push(`+${diff.edges.added}`);
  if (diff.edges.removed > 0) edgeChanges.push(`-${diff.edges.removed}`);
  if (diff.edges.modified > 0) edgeChanges.push(`~${diff.edges.modified}`);

  if (edgeChanges.length > 0) {
    parts.push(
      `${edgeChanges.join("/")} connection${diff.edges.added + diff.edges.removed + diff.edges.modified !== 1 ? "s" : ""}`,
    );
  }

  return parts.join(", ");
}

export function DraftRecoveryPopup({
  isInitialLoadComplete,
}: DraftRecoveryPopupProps) {
  const {
    isOpen,
    popupRef,
    nodeCount,
    edgeCount,
    diff,
    savedAt,
    onLoad,
    onDiscard,
  } = useDraftRecoveryPopup(isInitialLoadComplete);

  const diffSummary = formatDiffSummary(diff);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          ref={popupRef}
          className={cn("absolute left-1/2 top-4 z-50")}
          initial={{
            opacity: 0,
            x: "-50%",
            y: "-150%",
            scale: 0.5,
            filter: "blur(20px)",
          }}
          animate={{
            opacity: 1,
            x: "-50%",
            y: "0%",
            scale: 1,
            filter: "blur(0px)",
          }}
          exit={{
            opacity: 0,
            y: "-150%",
            scale: 0.5,
            filter: "blur(20px)",
            transition: { duration: 0.4, type: "spring", bounce: 0.2 },
          }}
          transition={{ duration: 0.2, type: "spring", bounce: 0.2 }}
        >
          <div
            className={cn(
              "flex items-center gap-3 rounded-xlarge border border-amber-200 bg-amber-50 px-4 py-3 shadow-lg",
            )}
          >
            <div className="flex items-center gap-2 text-amber-700 dark:text-amber-300">
              <ClockCounterClockwiseIcon className="h-5 w-5" weight="fill" />
            </div>

            <div className="flex flex-col">
              <Text
                variant="small-medium"
                className="text-amber-900 dark:text-amber-100"
              >
                Unsaved changes found
              </Text>
              <Text
                variant="small"
                className="text-amber-700 dark:text-amber-400"
              >
                {diffSummary ||
                  `${nodeCount} block${nodeCount !== 1 ? "s" : ""}, ${edgeCount} connection${edgeCount !== 1 ? "s" : ""}`}{" "}
                • {formatTimeAgo(new Date(savedAt).toISOString())}
              </Text>
            </div>

            <div className="ml-2 flex items-center gap-2">
              <Tooltip delayDuration={10}>
                <TooltipTrigger asChild>
                  <Button
                    variant="primary"
                    size="small"
                    onClick={onLoad}
                    className="aspect-square min-w-0 p-1.5"
                  >
                    <ClockCounterClockwiseIcon size={20} weight="fill" />
                    <span className="sr-only">Restore changes</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Restore changes</TooltipContent>
              </Tooltip>
              <Tooltip delayDuration={10}>
                <TooltipTrigger asChild>
                  <Button
                    variant="destructive"
                    size="icon"
                    onClick={onDiscard}
                    aria-label="Discard changes"
                    className="aspect-square min-w-0 p-1.5"
                  >
                    <XIcon size={20} />
                    <span className="sr-only">Discard changes</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Discard changes</TooltipContent>
              </Tooltip>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
