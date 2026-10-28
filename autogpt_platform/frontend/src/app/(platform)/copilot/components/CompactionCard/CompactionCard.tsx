"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { ReloadIcon } from "@hugeicons/core-free-icons";
import { formatElapsed } from "../JobStatsBar/formatElapsed";
import {
  barPercent,
  compactionLabel,
  type CompactionPhase,
  type CompactionStats,
} from "./helpers";
import { useCompactionProgress } from "./useCompactionProgress";
import { usePrefersReducedMotion } from "./usePrefersReducedMotion";

export const SHOW_TIME_AFTER_SECONDS = 20;

interface Props {
  phase: CompactionPhase | null;
  stats: CompactionStats;
  isSettled: boolean;
}

export function CompactionCard({ phase, stats, isSettled }: Props) {
  if (isSettled) {
    return (
      <div className="flex items-center gap-2 py-2 text-xs text-muted-foreground">
        <Icon icon={ReloadIcon} size={14} />
        <span className="min-w-0 truncate">
          {compactionLabel(phase, stats)}
        </span>
      </div>
    );
  }

  // A live row that hasn't received its first `data-compaction` part yet is
  // in the opening state — animate it as `summarizing` rather than letting
  // null read as finished copy over a frozen bar.
  return <LiveCompaction phase={phase ?? "summarizing"} stats={stats} />;
}

interface LiveProps {
  phase: CompactionPhase;
  stats: CompactionStats;
}

function LiveCompaction({ phase, stats }: LiveProps) {
  const { progress, elapsedSeconds } = useCompactionProgress(
    phase,
    stats.tokensBefore,
  );
  const prefersReducedMotion = usePrefersReducedMotion();
  const label = compactionLabel(phase, stats);
  const percent = Math.round(progress * 100);
  // Reduced motion gets the progress, not the crawl: the fill jumps a step at
  // a time instead of easing for minutes. `aria-valuenow` keeps the exact
  // percent either way — assistive tech reads the truth, not the rendering.
  const width = barPercent(progress, prefersReducedMotion);
  const showTime = elapsedSeconds >= SHOW_TIME_AFTER_SECONDS;

  return (
    <div className="flex flex-col gap-1.5 py-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Icon
          icon={ReloadIcon}
          size={14}
          className="[animation-duration:2.4s] motion-safe:animate-spin"
        />
        {/* Polite live region on the label ONLY — the progressbar's
            aria-valuenow moves every committed percent and would flood the
            announcement queue. */}
        <span aria-live="polite" className="min-w-0 flex-1 truncate">
          {label}
        </span>
        {showTime && (
          <span className="font-mono tabular-nums text-muted-foreground">
            {formatElapsed(elapsedSeconds)}
          </span>
        )}
      </div>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        className="h-1 w-full overflow-hidden rounded-full bg-muted"
      >
        <div
          className="h-full rounded-full bg-foreground"
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}
