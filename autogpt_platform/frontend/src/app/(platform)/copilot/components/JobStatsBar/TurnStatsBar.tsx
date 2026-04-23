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
 * Swap label between "Thought for X" and the formatted date on hover or click.
 * Hover = transient preview (desktop). Click = sticky toggle (touch).
 */
function TimeLabel({
  displaySeconds,
  localDate,
}: {
  displaySeconds: number;
  localDate: string | null;
}) {
  const [hovered, setHovered] = useState(false);
  const [clicked, setClicked] = useState(false);
  const labelText = `Thought for ${formatElapsed(displaySeconds)}`;

  if (!localDate) {
    return (
      <span className="text-[11px] tabular-nums text-neutral-500">
        {labelText}
      </span>
    );
  }

  const showDate = hovered || clicked;
  return (
    <button
      type="button"
      onClick={() => setClicked((c) => !c)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="cursor-pointer text-[11px] tabular-nums text-neutral-500"
    >
      {showDate ? localDate : labelText}
    </button>
  );
}

export function TurnStatsBar({ turnMessages, elapsedSeconds, stats }: Props) {
  const { counters } = getWorkDoneCounters(turnMessages);
  const displaySeconds = resolveDisplaySeconds(elapsedSeconds, stats);
  const localDate = stats?.createdAt ? formatLocalDate(stats.createdAt) : null;

  if (counters.length === 0 && displaySeconds === undefined) return null;

  return (
    <div className="mt-2 flex items-center gap-1.5">
      {displaySeconds !== undefined && (
        <TimeLabel displaySeconds={displaySeconds} localDate={localDate} />
      )}
      {counters.map(function renderCounter(counter, index) {
        const needsDot = index > 0 || displaySeconds !== undefined;
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
