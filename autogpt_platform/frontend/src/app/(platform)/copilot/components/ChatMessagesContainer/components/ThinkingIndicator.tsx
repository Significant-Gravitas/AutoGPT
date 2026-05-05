import { useEffect, useRef, useState } from "react";
import { formatElapsed } from "../../JobStatsBar/formatElapsed";
import { ScaleLoader } from "../../ScaleLoader/ScaleLoader";

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
   * "Contacting the model…", "Analyzing result…", "Optimizing conversation
   * context…"). When provided, it replaces the rotating generic phrase so
   * the user sees what's actually happening instead of a placeholder.
   */
  statusMessage?: string | null;
}

export function ThinkingIndicator({
  active,
  elapsedSeconds,
  statusMessage,
}: Props) {
  const { phrase, visible } = useCyclingPhrase(active);
  const showTime = active && elapsedSeconds >= SHOW_TIME_AFTER_SECONDS;
  const displayText = statusMessage || phrase;
  const transitionOpacity = statusMessage ? 1 : visible ? 1 : 0;

  return (
    <span className="inline-flex items-center gap-1.5 text-neutral-500">
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
