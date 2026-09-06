"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { StopIcon } from "@hugeicons/core-free-icons";

import type { VoiceState } from "../micStateMachine";
import { VoiceTrace, type TraceSource } from "./VoiceTrace";

/**
 * Each stage of the turn gets its own colour as well as its own motion:
 * green while the mic is live, amber while AutoPilot works, near-black
 * while it speaks. Colour is what makes the handover legible at a glance —
 * the shape alone read as one continuous animation.
 */
const APPEARANCE: Record<
  Exclude<VoiceState, "off">,
  { source: TraceSource; color: string }
> = {
  listening: { source: "mic", color: "bg-emerald-500" },
  hearing: { source: "mic", color: "bg-emerald-500" },
  transcribing: { source: "pulse", color: "bg-amber-500" },
  thinking: { source: "pulse", color: "bg-amber-500" },
  speaking: { source: "speech", color: "bg-zinc-900" },
};

interface Props {
  state: VoiceState;
  /** Read by screen readers only; sighted users get the trace instead. */
  statusLabel: string;
  onStop: () => void;
  /** Leaves voice mode and hands the composer back. */
  leaveButton?: React.ReactNode;
}

export function VoiceModeBar({
  state,
  statusLabel,
  onStop,
  leaveButton,
}: Props) {
  if (state === "off") return null;
  const { source, color } = APPEARANCE[state];

  return (
    <div className="flex w-full items-center gap-3 py-1.5 pl-3 pr-1.5">
      <VoiceTrace source={source} color={color} className="min-w-0 flex-1" />
      <span className="sr-only" role="status" aria-live="polite">
        {statusLabel}
      </span>
      {state === "speaking" && (
        <Button
          type="button"
          variant="secondary"
          size="small"
          onClick={onStop}
          aria-label="Stop speaking"
        >
          <Icon icon={StopIcon} className="mr-1 h-3.5 w-3.5" />
          Stop
        </Button>
      )}
      {leaveButton}
    </div>
  );
}
