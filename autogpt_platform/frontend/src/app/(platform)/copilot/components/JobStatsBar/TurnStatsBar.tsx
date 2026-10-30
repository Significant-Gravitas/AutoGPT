import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import type { TurnStats } from "../../helpers/convertChatSessionToUiMessages";
import { getTurnEndLabel, getTurnOutcome } from "./helpers";
import { getWorkDoneCounters } from "./useWorkDoneCounters";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  turnMessages: UIMessage<unknown, UIDataTypes, UITools>[];
  elapsedSeconds?: number;
  stats?: TurnStats;
  isLatest?: boolean;
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
 * Swap the label for the formatted date while the cursor is over it; revert
 * on mouse leave. Pure hover, no click toggle.
 */
function TimeLabel({
  label,
  localDate,
}: {
  label: string;
  localDate: string | null;
}) {
  const [hovered, setHovered] = useState(false);

  if (!localDate) return <span>{label}</span>;

  return (
    <span
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="cursor-default transition-colors hover:text-foreground-strong"
    >
      <span
        key={hovered ? "date" : "label"}
        className="inline-block duration-200 animate-in fade-in"
      >
        {hovered ? localDate : label}
      </span>
    </span>
  );
}

export function TurnStatsBar({
  turnMessages,
  elapsedSeconds,
  stats,
  isLatest = true,
}: Props) {
  const { counters } = getWorkDoneCounters(turnMessages);
  const displaySeconds = resolveDisplaySeconds(elapsedSeconds, stats);
  const localDate = stats?.createdAt ? formatLocalDate(stats.createdAt) : null;
  const outcome = isLatest ? getTurnOutcome(turnMessages) : "unsettled";

  const seconds =
    displaySeconds !== undefined && displaySeconds > 0 ? displaySeconds : null;
  if (counters.length === 0 && seconds === null) return null;
  const label = getTurnEndLabel(outcome, seconds);

  return (
    <div
      data-testid="turn-end-line"
      className={cn(
        "mt-2 flex items-center gap-1.5 text-caption-sm tabular-nums leading-inherit text-foreground-secondary",
        !isLatest && "opacity-50 transition-opacity group-hover:opacity-100",
      )}
    >
      {outcome === "done" && (
        <Icon
          icon={Tick02Icon}
          size={14}
          className="shrink-0 text-success"
          aria-hidden
        />
      )}
      {label !== null && <TimeLabel label={label} localDate={localDate} />}
      {counters.map(function renderCounter(counter, index) {
        const needsDot = index > 0 || label !== null;
        return (
          <span key={counter.category} className="flex items-center gap-1.5">
            {needsDot && (
              <Text variant="small-tight" as="span" tone="disabled" aria-hidden>
                &middot;
              </Text>
            )}
            <span>
              {counter.count} {counter.label}
            </span>
          </span>
        );
      })}
    </div>
  );
}
