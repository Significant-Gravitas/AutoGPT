import { useEffect, useRef, useState } from "react";
import { formatElapsed } from "../../JobStatsBar/formatElapsed";
import { PixelGridLoader } from "../../PixelGridLoader/PixelGridLoader";
import { ScaleLoader } from "../../ScaleLoader/ScaleLoader";
import { SwapText } from "../../ToolChain/SwapText";

const THINKING_PHRASES = [
  "Thinking...",
  "Considering this...",
  "Working through this...",
  "Analyzing your request...",
  "Reasoning...",
  "Looking into it...",
  "Processing your request...",
  "Mulling this over...",
  "Piecing it together...",
  "On it...",
  "Connecting the dots...",
  "Exploring possibilities...",
  "Weighing options...",
  "Diving deeper...",
  "Gathering thoughts...",
  "Almost there...",
  "Figuring this out...",
  "Putting it together...",
  "Running through ideas...",
  "Wrapping my head around this...",
];

const PHRASE_CYCLE_MS = 6_000;
const FADE_DURATION_MS = 300;

/** Only show elapsed time after this many seconds. */
const SHOW_TIME_AFTER_SECONDS = 20;

/**
 * Cycles through thinking phrases sequentially with a fade-out/in transition.
 * Returns the current phrase and whether it's visible (for opacity).
 */
function useCyclingPhrase(active: boolean) {
  const indexRef = useRef(0);
  const [phrase, setPhrase] = useState(THINKING_PHRASES[0]);
  const [visible, setVisible] = useState(true);
  const fadeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Reset to the first phrase when thinking restarts
  const prevActive = useRef(active);
  useEffect(() => {
    if (active && !prevActive.current) {
      indexRef.current = 0;
      setPhrase(THINKING_PHRASES[0]);
      setVisible(true);
    }
    prevActive.current = active;
  }, [active]);

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => {
      setVisible(false);
      fadeTimeoutRef.current = setTimeout(() => {
        indexRef.current = (indexRef.current + 1) % THINKING_PHRASES.length;
        setPhrase(THINKING_PHRASES[indexRef.current]);
        setVisible(true);
      }, FADE_DURATION_MS);
    }, PHRASE_CYCLE_MS);
    return () => {
      clearInterval(id);
      if (fadeTimeoutRef.current) {
        clearTimeout(fadeTimeoutRef.current);
        fadeTimeoutRef.current = null;
      }
    };
  }, [active]);

  return { phrase, visible };
}

interface Props {
  active: boolean;
  elapsedSeconds: number;
  /**
   * Backend-emitted status message for the current silent gap (e.g.
   * "Reading your message…", "Analyzing result…", "Optimizing conversation
   * context…"). When provided, it replaces the rotating generic phrase so
   * the user sees what's actually happening instead of a placeholder. In
   * the chain variant an absent message shows no label at all rather than
   * inventing a generic placeholder.
   */
  statusMessage?: string | null;
  /** "chain" is the NEW_TOOL_UI pixel-loader look; "legacy" (default)
   *  keeps the original rotating-phrase indicator. */
  variant?: "legacy" | "chain";
}

export function ThinkingIndicator({
  active,
  elapsedSeconds,
  statusMessage,
  variant = "legacy",
}: Props) {
  const { phrase, visible } = useCyclingPhrase(active);
  const showTime = active && elapsedSeconds >= SHOW_TIME_AFTER_SECONDS;

  if (variant === "chain") {
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

  const displayText = statusMessage || phrase;
  const transitionOpacity = statusMessage ? 1 : visible ? 1 : 0;

  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-neutral-500">
      <ScaleLoader size={16} />
      <span
        className="transition-opacity duration-300"
        style={{ opacity: transitionOpacity }}
      >
        <span className="animate-pulse [animation-duration:1.5s]">
          {displayText}
        </span>
      </span>
      {showTime && (
        <span className="animate-pulse tabular-nums [animation-duration:1.5s]">
          • {formatElapsed(elapsedSeconds)}
        </span>
      )}
    </span>
  );
}
