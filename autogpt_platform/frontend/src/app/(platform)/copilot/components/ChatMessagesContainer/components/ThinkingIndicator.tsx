import { formatElapsed } from "../../JobStatsBar/formatElapsed";
import { ScaleLoader } from "../../ScaleLoader/ScaleLoader";

/** Only show elapsed time after this many seconds. */
const SHOW_AFTER_SECONDS = 20;

interface Props {
  active: boolean;
  elapsedSeconds: number;
}

export function ThinkingIndicator({ active, elapsedSeconds }: Props) {
  const showTime = active && elapsedSeconds >= SHOW_AFTER_SECONDS;

  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-sm text-neutral-500">
      <ScaleLoader size={16} />
      {showTime && (
        <span className="tabular-nums">{formatElapsed(elapsedSeconds)}</span>
      )}
    </span>
  );
}
