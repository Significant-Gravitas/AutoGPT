import { useEffect, useRef, useState } from "react";
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
}

export function ThinkingIndicator({ active }: Props) {
  const { phrase, visible } = useCyclingPhrase(active);

  return (
    <span className="inline-flex items-center gap-1.5 text-neutral-500">
      <ScaleLoader size={16} />
      <span
        className="transition-opacity duration-300"
        style={{ opacity: visible ? 1 : 0 }}
      >
        <span className="animate-pulse [animation-duration:1.5s]">
          {phrase}
        </span>
      </span>
    </span>
  );
}
