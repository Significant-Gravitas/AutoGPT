"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { StopIcon } from "@hugeicons/core-free-icons";

import { isMicOpen, type VoiceState } from "../micStateMachine";
import { VoiceTrace } from "./VoiceTrace";

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
  const micOpen = isMicOpen(state);

  return (
    <div className="flex w-full items-center gap-3 py-1.5 pl-3 pr-1.5">
      <VoiceTrace
        source={micOpen ? "mic" : "pulse"}
        color={micOpen ? "bg-emerald-500" : "bg-zinc-400"}
        className="min-w-0 flex-1 justify-center"
      />
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
