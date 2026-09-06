"use client";

import type { VoiceState } from "../micStateMachine";
import { VoiceTrace, type TraceSource } from "./VoiceTrace";

/**
 * Each stage of the turn gets its own colour as well as its own motion:
 * green while the mic is live, the AutoGPT accent while AutoPilot works,
 * near-black while it speaks. Colour is what makes the handover legible at
 * a glance — the shape alone read as one continuous animation.
 */
const APPEARANCE: Record<
  Exclude<VoiceState, "off">,
  { source: TraceSource; color: string }
> = {
  listening: { source: "mic", color: "bg-emerald-500" },
  hearing: { source: "mic", color: "bg-emerald-500" },
  transcribing: { source: "pulse", color: "bg-accent" },
  thinking: { source: "pulse", color: "bg-accent" },
  speaking: { source: "speech", color: "bg-zinc-900" },
};

interface Props {
  state: VoiceState;
  /** Read by screen readers only; sighted users get the trace instead. */
  statusLabel: string;
  /** Leaves voice mode — and while AutoPilot speaks, is the stop control. */
  leaveButton?: React.ReactNode;
}

export function VoiceModeBar({ state, statusLabel, leaveButton }: Props) {
  if (state === "off") return null;
  const { source, color } = APPEARANCE[state];

  return (
    <div className="flex w-full items-center gap-3 py-1.5 pl-3 pr-1.5">
      <VoiceTrace source={source} color={color} className="min-w-0 flex-1" />
      <span className="sr-only" role="status" aria-live="polite">
        {statusLabel}
      </span>
      {leaveButton}
    </div>
  );
}
