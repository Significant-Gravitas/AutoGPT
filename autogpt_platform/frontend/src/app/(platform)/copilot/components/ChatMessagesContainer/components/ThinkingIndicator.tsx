import { formatElapsed } from "../../JobStatsBar/formatElapsed";
import { PixelGridLoader } from "../../PixelGridLoader/PixelGridLoader";
import { SwapText } from "../../ToolChain/SwapText";

/** Only show elapsed time after this many seconds. */
const SHOW_TIME_AFTER_SECONDS = 20;

interface Props {
  active: boolean;
  elapsedSeconds: number;
  /**
   * Backend-emitted status message for the current silent gap (e.g.
   * "Reading your message…", "Analyzing result…", "Optimizing conversation
   * context…"). Absent means we genuinely don't know what's happening, so
   * no label is shown at all rather than inventing a generic placeholder.
   */
  statusMessage?: string | null;
  /** Overrides `SHOW_TIME_AFTER_SECONDS`; 0 keeps the timer always visible. */
  showTimeAfterSeconds?: number;
}

export function ThinkingIndicator({
  active,
  elapsedSeconds,
  statusMessage,
  showTimeAfterSeconds = SHOW_TIME_AFTER_SECONDS,
}: Props) {
  const showTime = active && elapsedSeconds >= showTimeAfterSeconds;

  return (
    <span className="inline-flex w-fit items-center gap-2.5 text-sm text-zinc-600">
      <PixelGridLoader className="text-zinc-600" />
      {statusMessage && (
        <SwapText text={statusMessage} shimmer className="text-sm" />
      )}
      {showTime && (
        <span className="font-mono text-xs tabular-nums text-zinc-400">
          {formatElapsed(elapsedSeconds)}
        </span>
      )}
    </span>
  );
}
