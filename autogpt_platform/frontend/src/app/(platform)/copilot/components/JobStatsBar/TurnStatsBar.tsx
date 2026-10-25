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
 * The saved whole-turn durationMs once present, so a reload reads the same;
 * the live count, which runs long while a background tab types, until then.
 */
function resolveDisplaySeconds(
  elapsedSeconds: number | undefined,
  stats: TurnStats | undefined,
): number | undefined {
  if (stats?.durationMs !== undefined) {
    return Math.round(stats.durationMs / 1000);
  }
  if (elapsedSeconds !== undefined && elapsedSeconds > 0) return elapsedSeconds;
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
      <span
        // TODO(tokens): no exact role for text-neutral-500
        className="text-caption-sm tabular-nums leading-inherit text-neutral-500"
      >
        {labelText}
      </span>
    );
  }

  return (
    <span
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      // TODO(tokens): no exact role for text-neutral-500, hover:text-neutral-700
      className="cursor-default text-caption-sm tabular-nums leading-inherit text-neutral-500 transition-colors hover:text-neutral-700"
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
    <div className="mt-2 flex items-center gap-1.5 text-xs opacity-50 transition-opacity group-hover:opacity-100">
      {showTimeLabel !== null && (
        <TimeLabel displaySeconds={showTimeLabel} localDate={localDate} />
      )}
      {counters.map(function renderCounter(counter, index) {
        const needsDot = index > 0 || showTimeLabel !== null;
        return (
          <span key={counter.category} className="flex items-center gap-1">
            {needsDot && (
              <span
                // TODO(tokens): no exact role for text-neutral-300
                className="text-xs text-neutral-300"
              >
                &middot;
              </span>
            )}
            <span
              // TODO(tokens): no exact role for text-neutral-500
              className="text-caption-sm tabular-nums leading-inherit text-neutral-500"
            >
              {counter.count} {counter.label}
            </span>
          </span>
        );
      })}
    </div>
  );
}
