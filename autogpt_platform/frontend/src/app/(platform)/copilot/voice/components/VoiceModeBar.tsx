"use client";

import { Button } from "@/components/atoms/Button/Button";

import type { VoiceState } from "../micStateMachine";
import type { VoiceFailure } from "../useVoiceMode";
import { VoiceTrace, type TraceSource } from "./VoiceTrace";

/**
 * Each stage of the turn gets its own colour as well as its own motion:
 * green while the mic is live, the AutoGPT accent while Otto works,
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
  /** Leaves voice mode — and while Otto speaks, is the stop control. */
  leaveButton?: React.ReactNode;
  /** A transcription that failed with the audio still in hand. */
  failure?: VoiceFailure | null;
  onRetry?: () => void;
  onDownload?: () => void;
}

export function VoiceModeBar({
  state,
  statusLabel,
  leaveButton,
  failure = null,
  onRetry,
  onDownload,
}: Props) {
  if (state === "off") return null;

  // The trace says "everything is fine"; a failure has to take its place
  // rather than sit beside it, and it stays until the user acts on it.
  if (failure) {
    return (
      <div className="flex w-full flex-wrap items-center gap-2 py-1.5 pl-3 pr-1.5">
        <span
          role="alert"
          className="min-w-0 flex-1 truncate text-sm text-red-600"
        >
          {failure.message}
        </span>
        <Button type="button" variant="ghost" size="xs" onClick={onRetry}>
          Retry
        </Button>
        <Button type="button" variant="ghost" size="xs" onClick={onDownload}>
          Download recording
        </Button>
        {leaveButton}
      </div>
    );
  }

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
