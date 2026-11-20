import { useNowSecond } from "@/hooks/useNowSecond";
import { SHOW_TIME_AFTER_SECONDS } from "../../JobStatsBar/constants";
import { formatElapsed } from "../../JobStatsBar/formatElapsed";
import { getElapsedSeconds } from "../../JobStatsBar/useElapsedTimer";
import { StoneStackLoader } from "../../StoneStackLoader/StoneStackLoader";
import { SwapText } from "../../ToolChain/SwapText";

interface Props {
  active: boolean;
  /**
   * Epoch ms the wait started. When set, the indicator runs its own clock
   * (the shared `useNowSecond` ticker, only while `active`), so the parent
   * does not re-render every second.
   */
  startedAtMs?: number | null;
  /** Elapsed seconds from a parent that already owns a clock. */
  elapsedSeconds?: number;
  /**
   * Plain-language label for a wait the UI itself narrates (e.g. "Uploading
   * 2 files…"). Defaults to "Thinking…". The backend's own `data-status`
   * lines ("Restoring your session…", "Preparing workspace…") never reach
   * here: they describe plumbing, not anything the user needs to know.
   */
  statusMessage?: string | null;
  /**
   * The material colour of whoever is thinking (Otto's lavender, an expert's
   * own). Leave it out for a wait nobody is thinking through, such as an
   * upload or a reconnect: the loader then stays the grey of the label.
   */
  color?: string | null;
}

export function ThinkingIndicator({
  active,
  startedAtMs,
  elapsedSeconds = 0,
  statusMessage,
  color,
}: Props) {
  const hasClock = startedAtMs !== null && startedAtMs !== undefined;
  const nowMs = useNowSecond(active && hasClock);
  const seconds = hasClock
    ? getElapsedSeconds(startedAtMs, nowMs)
    : elapsedSeconds;
  const showTime = active && seconds >= SHOW_TIME_AFTER_SECONDS;

  return (
    <span
      className="inline-flex w-fit items-center gap-2.5 text-body text-foreground-secondary"
      role="status"
      aria-live="polite"
    >
      <StoneStackLoader size={16} color={color} />
      <SwapText
        text={statusMessage ?? "Thinking…"}
        shimmer
        className="text-body"
      />
      {showTime && (
        <span className="font-mono text-caption text-foreground-subtle tabular-nums">
          {formatElapsed(seconds)}
        </span>
      )}
    </span>
  );
}
