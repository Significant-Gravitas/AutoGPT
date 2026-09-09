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
   * context…"). An absent message shows no label at all rather than
   * inventing a generic placeholder.
   */
  statusMessage?: string | null;
}

export function ThinkingIndicator({
  active,
  elapsedSeconds,
  statusMessage,
}: Props) {
  const showTime = active && elapsedSeconds >= SHOW_TIME_AFTER_SECONDS;

  return (
    <span
      role="status"
      aria-live="polite"
      className="inline-flex w-fit items-center gap-2.5 text-sm text-zinc-600"
    >
      <PixelGridLoader className="text-zinc-600" />
      {statusMessage && (
        <SwapText text={statusMessage} shimmer className="text-sm" />
      )}
      {!statusMessage && <span className="sr-only">Thinking…</span>}
      {showTime && (
        <span className="font-mono text-xs tabular-nums text-zinc-400">
          {formatElapsed(elapsedSeconds)}
        </span>
      )}
    </span>
  );
}
