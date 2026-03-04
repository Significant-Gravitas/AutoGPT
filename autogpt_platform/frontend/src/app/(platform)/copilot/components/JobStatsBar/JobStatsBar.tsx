import { cn } from "@/lib/utils";
import { TimerIcon } from "@phosphor-icons/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { ToolInvocationsDialog } from "./ToolInvocationsDialog";
import { useJobTimer } from "./useJobTimer";
import {
  useWorkDoneCounters,
  type WorkDoneCounter,
} from "./useWorkDoneCounters";

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  /** Chat status from the Vercel AI SDK (e.g. "streaming", "submitted", "ready"). */
  status: string;
}

export function JobStatsBar({ messages, status }: Props) {
  const isActive = status === "streaming" || status === "submitted";

  const { formattedTime, isRunning, hasStarted } = useJobTimer({
    isActive,
  });
  const { counters } = useWorkDoneCounters(messages);
  const [activeCounter, setActiveCounter] = useState<WorkDoneCounter | null>(
    null,
  );

  // Don't render anything if the timer has never started
  if (!hasStarted) {
    return null;
  }

  const hasCounters = counters.length > 0;

  return (
    <>
      <div
        role="status"
        aria-live="polite"
        className={cn(
          "flex items-center justify-center gap-1.5 px-3 py-1.5",
          "transition-opacity duration-300",
          isRunning ? "opacity-100" : "opacity-80",
        )}
      >
        <div className="flex items-center gap-1.5">
          {/* Work-done counters */}
          {counters.map(function renderCounter(counter, index) {
            return (
              <span key={counter.category} className="flex items-center gap-1">
                {index > 0 && (
                  <span className="text-xs text-neutral-300">&middot;</span>
                )}
                <button
                  type="button"
                  onClick={() => setActiveCounter(counter)}
                  className={cn(
                    "text-[11px] tabular-nums text-neutral-500 underline decoration-dotted underline-offset-2 hover:text-neutral-700",
                    isRunning && "transition-all duration-200",
                  )}
                >
                  {counter.count} {counter.label}
                </button>
              </span>
            );
          })}

          {/* Separator between counters and timer */}
          {hasCounters && (
            <span className="text-xs text-neutral-300">&middot;</span>
          )}

          {/* Timer */}
          <span className="flex items-center gap-1">
            <TimerIcon
              size={12}
              weight={isRunning ? "fill" : "regular"}
              className={cn(
                "shrink-0 text-neutral-400",
                isRunning && "animate-pulse",
              )}
            />
            <span
              className={cn(
                "text-[11px] tabular-nums",
                isRunning ? "text-neutral-500" : "text-neutral-400",
              )}
            >
              {isRunning ? formattedTime : `Completed in ${formattedTime}`}
            </span>
          </span>
        </div>
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
