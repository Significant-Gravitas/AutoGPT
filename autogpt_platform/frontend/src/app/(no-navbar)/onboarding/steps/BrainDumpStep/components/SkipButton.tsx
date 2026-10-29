"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";

interface Props {
  isHidden: boolean;
  isProcessing: boolean;
  onSkip: () => void;
}

// Pinned to the page's top right corner, so it is rendered outside the
// step's own positioned wrapper.
export function SkipButton({ isHidden, isProcessing, onSkip }: Props) {
  return (
    <div
      className={cn(
        "absolute right-4 top-4 flex items-center gap-2 sm:right-6 sm:top-6 sm:gap-5",
        isHidden && "hidden",
      )}
    >
      {/* Skipping mid-submit would advance the wizard a second time
          behind the finalize that is already in flight, landing past
          the last step on a blank screen. */}
      {!isProcessing && (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={onSkip}
          className="text-foreground-subtle hover:text-foreground-body"
        >
          Skip for now
        </Button>
      )}
    </div>
  );
}
