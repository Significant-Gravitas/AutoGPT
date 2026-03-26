import type { UIDataTypes, UIMessage, UITools } from "ai";
import { formatElapsed } from "./formatElapsed";
import { getWorkDoneCounters } from "./useWorkDoneCounters";

interface Props {
  turnMessages: UIMessage<unknown, UIDataTypes, UITools>[];
  elapsedSeconds?: number;
}

export function TurnStatsBar({ turnMessages, elapsedSeconds }: Props) {
  const { counters } = getWorkDoneCounters(turnMessages);
  const hasTime = elapsedSeconds !== undefined && elapsedSeconds > 0;

  if (counters.length === 0 && !hasTime) return null;

  return (
    <div className="mt-2 flex items-center gap-1.5">
      {hasTime && (
        <span className="text-[11px] tabular-nums text-neutral-500">
          Thought for {formatElapsed(elapsedSeconds)}
        </span>
      )}
      {counters.map(function renderCounter(counter, index) {
        const needsDot = index > 0 || hasTime;
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
  );
}
