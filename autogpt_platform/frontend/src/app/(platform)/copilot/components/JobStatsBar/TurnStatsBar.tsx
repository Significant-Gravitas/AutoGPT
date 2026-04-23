import type { UIDataTypes, UIMessage, UITools } from "ai";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { formatElapsed } from "./formatElapsed";
import { getWorkDoneCounters } from "./useWorkDoneCounters";

interface Props {
  turnMessages: UIMessage<unknown, UIDataTypes, UITools>[];
  elapsedSeconds?: number;
  durationMs?: number;
  reasoningDurationMs?: number;
  timestamp?: string;
}

function formatLocalTimestamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  });
}

export function TurnStatsBar({
  turnMessages,
  elapsedSeconds,
  durationMs,
  reasoningDurationMs,
  timestamp,
}: Props) {
  const { counters } = getWorkDoneCounters(turnMessages);

  // Prefer live elapsedSeconds while streaming.  Once the turn is finalized
  // use reasoningDurationMs when the backend recorded actual reasoning time
  // — it excludes tool execution, which the user perceives as dead time.
  // Fall back to the whole-turn wall clock for older turns that never had
  // a reasoningDurationMs recorded.
  let displaySeconds: number | undefined;
  if (elapsedSeconds !== undefined && elapsedSeconds > 0) {
    displaySeconds = elapsedSeconds;
  } else if (reasoningDurationMs !== undefined && reasoningDurationMs > 0) {
    displaySeconds = Math.max(1, Math.round(reasoningDurationMs / 1000));
  } else if (durationMs !== undefined && durationMs > 0) {
    displaySeconds = Math.round(durationMs / 1000);
  }

  const hasTime = displaySeconds !== undefined && displaySeconds > 0;
  const localTime = timestamp ? formatLocalTimestamp(timestamp) : null;

  if (counters.length === 0 && !hasTime && !localTime) return null;

  const timeLabel = hasTime ? (
    <span className="cursor-default text-[11px] tabular-nums text-neutral-500">
      Thought for {formatElapsed(displaySeconds!)}
    </span>
  ) : null;

  return (
    <TooltipProvider>
      <div className="mt-2 flex items-center gap-1.5">
        {timeLabel &&
          (localTime ? (
            <Tooltip>
              <TooltipTrigger asChild>{timeLabel}</TooltipTrigger>
              <TooltipContent side="top">{localTime}</TooltipContent>
            </Tooltip>
          ) : (
            timeLabel
          ))}
        {!hasTime && localTime && (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-default text-[11px] tabular-nums text-neutral-500">
                {localTime}
              </span>
            </TooltipTrigger>
            <TooltipContent side="top">{localTime}</TooltipContent>
          </Tooltip>
        )}
        {counters.map(function renderCounter(counter, index) {
          const needsDot = index > 0 || hasTime || !!localTime;
          return (
            <span key={counter.category} className="flex items-center gap-1">
              {needsDot && (
                <span className="text-xs text-neutral-300">&middot;</span>
              )}
              <span className="text-[11px] tabular-nums text-neutral-500">
                {counter.count} {counter.label}
              </span>
            </span>
          );
        })}
      </div>
    </TooltipProvider>
  );
}
