import { useEffect, useRef, useState } from "react";

const CHAR_DELAY_MS = 18;
const START_DELAY_MS = 250;

export function useTypewriter(
  text: string,
  onComplete?: () => void,
  enabled = true,
) {
  const [typedCount, setTypedCount] = useState(0);
  const onCompleteRef = useRef(onComplete);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    if (!enabled || prefersReducedMotion()) {
      setTypedCount(text.length);
      onCompleteRef.current?.();
      return;
    }

    setTypedCount(0);
    let index = 0;
    let timer: ReturnType<typeof setTimeout>;

    function type() {
      index += 1;
      setTypedCount(index);
      if (index < text.length) {
        timer = setTimeout(type, CHAR_DELAY_MS);
        return;
      }
      onCompleteRef.current?.();
    }

    timer = setTimeout(type, START_DELAY_MS);
    return () => clearTimeout(timer);
  }, [text, enabled]);

  return {
    typed: text.slice(0, typedCount),
    isTyping: typedCount < text.length,
  };
}

function prefersReducedMotion() {
  if (typeof window === "undefined") return true;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
