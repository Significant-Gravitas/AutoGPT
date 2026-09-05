"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { StopIcon } from "@hugeicons/core-free-icons";

import type { VoiceState } from "../micStateMachine";

interface Props {
  state: VoiceState;
  statusLabel: string;
  onStop: () => void;
}

export function VoiceModeBar({ state, statusLabel, onStop }: Props) {
  if (state === "off") return null;

  return (
    <div
      className="flex items-center justify-center gap-3 py-2"
      role="status"
      aria-live="polite"
    >
      <span
        className={cn(
          "size-2 rounded-full bg-zinc-400",
          state === "hearing" && "animate-pulse bg-red-500",
          state === "listening" && "bg-emerald-500",
          state === "speaking" && "animate-pulse bg-zinc-900",
        )}
      />
      <span className="text-sm text-zinc-600">{statusLabel}</span>
      {state === "speaking" && (
        <Button type="button" variant="secondary" size="small" onClick={onStop}>
          <Icon icon={StopIcon} className="mr-1 h-3.5 w-3.5" />
          Stop
        </Button>
      )}
    </div>
  );
}
