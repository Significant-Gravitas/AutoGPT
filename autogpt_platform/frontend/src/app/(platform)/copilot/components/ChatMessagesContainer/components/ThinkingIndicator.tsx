import { formatElapsed } from "../../JobStatsBar/formatElapsed";
import { PixelGridLoader } from "../../PixelGridLoader/PixelGridLoader";
import { SwapText } from "../../ToolChain/SwapText";

/** Only show elapsed time after this many seconds. */
const SHOW_TIME_AFTER_SECONDS = 20;

interface Props {
  active: boolean;
  elapsedSeconds: number;
  /**
   * Plain-language label for a wait the UI itself narrates (e.g. "Uploading
   * 2 files…"). Defaults to "Thinking…". The backend's own `data-status`
   * lines ("Restoring your session…", "Preparing workspace…") never reach
   * here: they describe plumbing, not anything the user needs to know.
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
      <SwapText
        text={statusMessage ?? "Thinking…"}
        shimmer
        className="text-sm"
      />
      {showTime && (
        <span className="font-mono text-xs tabular-nums text-zinc-400">
          {formatElapsed(elapsedSeconds)}
        </span>
      )}
    </span>
  );
}
