"use client";

import { RepeatMenu } from "@/app/(platform)/tasks/components/RepeatMenu/RepeatMenu";
import { Button } from "@/components/atoms/Button/Button";
import { PlayIcon } from "@hugeicons/core-free-icons";

interface Props {
  title: string;
  isPaused: boolean;
  /** Started by an event, so it has no Run now and stops watching. */
  isWatching?: boolean;
  isStarting: boolean;
  isBusy: boolean;
  onRunNow: () => void;
  onPause: () => void;
  onResume: () => void;
  onStopRepeating: () => void;
  onCancel: () => void;
}

/** The header of a repeating or watching task between runs: run it now (a
 *  watching task runs when its event comes), pause or resume it, and a menu
 *  to stop it repeating or watching, or cancel it. */
export function RepeatControls({
  title,
  isPaused,
  isWatching = false,
  isStarting,
  isBusy,
  onRunNow,
  onPause,
  onResume,
  onStopRepeating,
  onCancel,
}: Props) {
  return (
    <div className="flex items-center gap-2">
      {isWatching ? null : (
        <Button
          variant="primary"
          size="xs"
          leadingIcon={PlayIcon}
          loading={isStarting}
          disabled={isBusy}
          onClick={onRunNow}
        >
          Run now
        </Button>
      )}
      <Button
        variant="secondary"
        size="xs"
        disabled={isBusy}
        onClick={isPaused ? onResume : onPause}
      >
        {isPaused ? "Resume" : "Pause"}
      </Button>
      <RepeatMenu
        title={title}
        isBusy={isBusy}
        isWatching={isWatching}
        onStopRepeating={onStopRepeating}
        onCancel={onCancel}
      />
    </div>
  );
}
