import { useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";

// About as long as it takes to read "Otto is typing" before the reply lands.
export const OTTO_TYPING_MS = 900;

// True while Otto's typing dots show. Each mount types once; with reduced
// motion there is nothing to wait for.
export function useOttoTyping() {
  const reduceMotion = Boolean(useReducedMotion());
  const [isWaiting, setIsWaiting] = useState(true);

  useEffect(() => {
    if (reduceMotion) return;
    const timer = setTimeout(() => setIsWaiting(false), OTTO_TYPING_MS);
    return () => clearTimeout(timer);
  }, [reduceMotion]);

  return { isTyping: isWaiting && !reduceMotion, reduceMotion };
}
