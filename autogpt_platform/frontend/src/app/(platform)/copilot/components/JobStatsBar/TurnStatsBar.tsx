import type { UIDataTypes, UIMessage, UITools } from "ai";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import type { TurnStats } from "../../helpers/convertChatSessionToUiMessages";
import { formatElapsed } from "./formatElapsed";
import { getWorkDoneCounters } from "./useWorkDoneCounters";

interface Props {
  turnMessages: UIMessage<unknown, UIDataTypes, UITools>[];
  elapsedSeconds?: number;
  stats?: TurnStats;
}

function formatLocalTimestamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  });
}

/**
 * Prefer live elapsedSeconds while streaming. Once the turn is finalized use
 * reasoningDurationMs when the backend recorded actual reasoning time — it
 * excludes tool execution, which the user perceives as dead time. Fall back
 * to the whole-turn wall clock for legacy rows that never had a
 * reasoningDurationMs recorded.
 */
function resolveDisplaySeconds(
  elapsedSeconds: number | undefined,
  stats: TurnStats | undefined,
): number | undefined {
  if (elapsedSeconds !== undefined && elapsedSeconds > 0) return elapsedSeconds;
  if (stats?.reasoningDurationMs && stats.reasoningDurationMs > 0) {
    return Math.max(1, Math.round(stats.reasoningDurationMs / 1000));
  }
  if (stats?.durationMs && stats.durationMs > 0) {
    return Math.round(stats.durationMs / 1000);
  }
  return undefined;
}

export function TurnStatsBar({ turnMessages, elapsedSeconds, stats }: Props) {
  const { counters } = getWorkDoneCounters(turnMessages);
  const displaySeconds = resolveDisplaySeconds(elapsedSeconds, stats);
  const localTime = stats?.createdAt
    ? formatLocalTimestamp(stats.createdAt)
    : null;

  if (counters.length === 0 && displaySeconds === undefined && !localTime)
    return null;

  return (
    <TooltipProvider>
      <div className="mt-2 flex items-center gap-1.5">
        {displaySeconds !== undefined &&
          (localTime ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="cursor-default text-[11px] tabular-nums text-neutral-500">
                  Thought for {formatElapsed(displaySeconds)}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">{localTime}</TooltipContent>
            </Tooltip>
          ) : (
            <span className="text-[11px] tabular-nums text-neutral-500">
              Thought for {formatElapsed(displaySeconds)}
            </span>
          ))}
        {displaySeconds === undefined && localTime && (
          <span className="text-[11px] tabular-nums text-neutral-500">
            {localTime}
          </span>
        )}
        {counters.map(function renderCounter(counter, index) {
          const needsDot =
            index > 0 || displaySeconds !== undefined || !!localTime;
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
