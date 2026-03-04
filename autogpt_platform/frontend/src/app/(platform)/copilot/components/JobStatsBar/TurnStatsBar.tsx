import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useWorkDoneCounters } from "./useWorkDoneCounters";

interface Props {
  /** Messages scoped to this turn (user message + assistant response) */
  turnMessages: UIMessage<unknown, UIDataTypes, UITools>[];
}

export function TurnStatsBar({ turnMessages }: Props) {
  const { counters } = useWorkDoneCounters(turnMessages);

  if (counters.length === 0) return null;

  return (
    <div className="mt-2 flex items-center gap-1.5">
      {counters.map(function renderCounter(counter, index) {
        return (
          <span key={counter.category} className="flex items-center gap-1">
            {index > 0 && (
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
