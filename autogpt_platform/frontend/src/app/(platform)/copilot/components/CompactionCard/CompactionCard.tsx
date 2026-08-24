"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ReloadIcon } from "@hugeicons/core-free-icons";
import { formatElapsed } from "../JobStatsBar/formatElapsed";
import {
  compactionLabel,
  type CompactionPhase,
  type CompactionStats,
} from "./helpers";
import { useCompactionProgress } from "./useCompactionProgress";

const SHOW_TIME_AFTER_SECONDS = 20;

interface Props {
  phase: CompactionPhase | null;
  stats: CompactionStats;
  isSettled: boolean;
}

export function CompactionCard({ phase, stats, isSettled }: Props) {
  const label = compactionLabel(phase, stats);

  if (isSettled) {
    return (
      <div className="flex items-center gap-2 py-2 text-xs text-muted-foreground">
        <Icon icon={ReloadIcon} size={14} />
        <span className="min-w-0 truncate">{label}</span>
      </div>
    );
  }

  return <LiveCompaction phase={phase} stats={stats} label={label} />;
}

interface LiveProps {
  phase: CompactionPhase | null;
  stats: CompactionStats;
  label: string;
}

function LiveCompaction({ phase, stats, label }: LiveProps) {
  const { progress, elapsedSeconds } = useCompactionProgress(
    phase,
    stats.tokensBefore,
  );
  const percent = Math.round(progress * 100);
  const showTime = elapsedSeconds >= SHOW_TIME_AFTER_SECONDS;

  return (
    <div className="flex flex-col gap-1.5 py-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Icon
          icon={ReloadIcon}
          size={14}
          className={cn(
            phase !== "done" && "animate-spin [animation-duration:2.4s]",
          )}
        />
        <span className="min-w-0 flex-1 truncate">{label}</span>
        {showTime && phase !== "done" && (
          <span className="font-mono tabular-nums text-zinc-400">
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
        className="h-1 w-full overflow-hidden rounded-full bg-zinc-200"
      >
        <div
          className="h-full rounded-full bg-zinc-900"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
