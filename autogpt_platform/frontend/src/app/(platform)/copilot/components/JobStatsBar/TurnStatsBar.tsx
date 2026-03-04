import { TimerIcon } from "@phosphor-icons/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { ToolInvocationsDialog } from "./ToolInvocationsDialog";
import {
  useWorkDoneCounters,
  type WorkDoneCounter,
} from "./useWorkDoneCounters";

interface Props {
  /** Messages scoped to this turn (user message + assistant response) */
  turnMessages: UIMessage<unknown, UIDataTypes, UITools>[];
  /** Duration in ms from backend, or null if not yet available */
  durationMs: number | null;
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

export function TurnStatsBar({ turnMessages, durationMs }: Props) {
  const { counters } = useWorkDoneCounters(turnMessages);
  const [activeCounter, setActiveCounter] = useState<WorkDoneCounter | null>(
    null,
  );

  if (counters.length === 0 && durationMs == null) return null;

  return (
    <>
      <div className="mt-2 flex items-center gap-1.5">
        {counters.map(function renderCounter(counter, index) {
          return (
            <span key={counter.category} className="flex items-center gap-1">
              {index > 0 && (
                <span className="text-xs text-neutral-300">&middot;</span>
              )}
              <button
                type="button"
                onClick={() => setActiveCounter(counter)}
                className="text-[11px] tabular-nums text-neutral-500 underline decoration-dotted underline-offset-2 hover:text-neutral-700"
              >
                {counter.count} {counter.label}
              </button>
            </span>
          );
        })}

        {counters.length > 0 && durationMs != null && (
          <span className="text-xs text-neutral-300">&middot;</span>
        )}

        {durationMs != null && (
          <span className="flex items-center gap-1">
            <TimerIcon size={12} className="shrink-0 text-neutral-400" />
            <span className="text-[11px] tabular-nums text-neutral-400">
              Completed in {formatDuration(durationMs)}
            </span>
          </span>
        )}
      </div>

      {activeCounter && (
        <ToolInvocationsDialog
          title={`${activeCounter.count} ${activeCounter.label}`}
          invocations={activeCounter.invocations}
          onClose={() => setActiveCounter(null)}
        />
      )}
    </>
  );
}
