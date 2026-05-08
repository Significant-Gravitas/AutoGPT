import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import type { TurnStats } from "../../helpers/convertChatSessionToUiMessages";
import { formatElapsed } from "./formatElapsed";
import { getWorkDoneCounters } from "./useWorkDoneCounters";

interface Props {
  turnMessages: UIMessage<unknown, UIDataTypes, UITools>[];
  elapsedSeconds?: number;
  stats?: TurnStats;
}

function formatLocalDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

/**
 * Prefer live elapsedSeconds while streaming; fall back to the persisted
 * whole-turn durationMs afterwards.
 */
function resolveDisplaySeconds(
  elapsedSeconds: number | undefined,
  stats: TurnStats | undefined,
): number | undefined {
  if (elapsedSeconds !== undefined && elapsedSeconds > 0) return elapsedSeconds;
  if (stats?.durationMs && stats.durationMs > 0) {
    return Math.round(stats.durationMs / 1000);
  }
  return undefined;
}

/**
 * Swap "Thought for X" → the formatted date while the cursor is over the
 * label; revert on mouse leave.  Pure hover, no click toggle.
 */
function TimeLabel({
  displaySeconds,
  localDate,
}: {
  displaySeconds: number;
  localDate: string | null;
}) {
  const [hovered, setHovered] = useState(false);
  const labelText = `Thought for ${formatElapsed(displaySeconds)}`;

  if (!localDate) {
    return (
      <span className="text-[11px] tabular-nums text-neutral-500">
        {labelText}
      </span>
    );
  }

  return (
    <span
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="cursor-default text-[11px] tabular-nums text-neutral-500 transition-colors hover:text-neutral-700"
    >
      <span
        key={hovered ? "date" : "label"}
        className="inline-block duration-200 animate-in fade-in"
      >
        {hovered ? localDate : labelText}
      </span>
    </span>
  );
}

export function TurnStatsBar({ turnMessages, elapsedSeconds, stats }: Props) {
  const { counters } = getWorkDoneCounters(turnMessages);
  const displaySeconds = resolveDisplaySeconds(elapsedSeconds, stats);
  const localDate = stats?.createdAt ? formatLocalDate(stats.createdAt) : null;

  const showTimeLabel =
    displaySeconds !== undefined && displaySeconds > 0 ? displaySeconds : null;
  if (counters.length === 0 && showTimeLabel === null) return null;

  return (
    <div className="mt-2 flex items-center gap-1.5">
      {showTimeLabel !== null && (
        <TimeLabel displaySeconds={showTimeLabel} localDate={localDate} />
      )}
      {counters.map(function renderCounter(counter, index) {
        const needsDot = index > 0 || showTimeLabel !== null;
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
