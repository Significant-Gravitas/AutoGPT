"use client";

import { Button } from "@/components/atoms/Button/Button";
import type { ScreenState } from "../useBrainDumpStep";

interface Props {
  screen: ScreenState;
  isPaused: boolean;
  onToggle: () => void;
}

const ROTATING_ON: ScreenState[] = ["rest", "typing", "recording"];

// The keyboard route to pausing the ideas. A mouse or touch user clicks an
// idea instead, so this stays out of sight until it takes focus.
export function IdeasPauseToggle({ screen, isPaused, onToggle }: Props) {
  if (!ROTATING_ON.includes(screen)) return null;
  return (
    <Button
      type="button"
      variant="secondary"
      size="xs"
      aria-pressed={isPaused}
      onClick={onToggle}
      className="sr-only focus-visible:not-sr-only focus-visible:absolute focus-visible:left-1/2 focus-visible:top-0 focus-visible:z-20 focus-visible:-translate-x-1/2"
    >
      Pause changing ideas
    </Button>
  );
}
