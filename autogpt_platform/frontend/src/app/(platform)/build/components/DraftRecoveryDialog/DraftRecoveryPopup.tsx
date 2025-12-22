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

interface DraftRecoveryPopupProps {
  isInitialLoadComplete: boolean;
}

export function DraftRecoveryPopup({
  isInitialLoadComplete,
}: DraftRecoveryPopupProps) {
  const { isOpen, popupRef, nodeCount, edgeCount, savedAt, onLoad, onDiscard } =
    useDraftRecoveryPopup(isInitialLoadComplete);

  if (!isOpen) return null;

  return (
    <div
      ref={popupRef}
      className={cn(
        "absolute left-1/2 top-4 z-50 -translate-x-1/2",
        "duration-200 animate-in fade-in-0 slide-in-from-top-2",
      )}
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
          <span className="text-sm font-medium text-amber-900 dark:text-amber-100">
            Unsaved changes found
          </span>
          <span className="text-xs text-amber-700 dark:text-amber-400">
            {nodeCount} block{nodeCount !== 1 ? "s" : ""}, {edgeCount}{" "}
            connection
            {edgeCount !== 1 ? "s" : ""} •{" "}
            {formatTimeAgo(new Date(savedAt).toISOString())}
          </span>
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
    </div>
  );
}
