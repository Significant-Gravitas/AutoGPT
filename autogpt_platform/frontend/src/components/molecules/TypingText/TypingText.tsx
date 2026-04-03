"use client";

import { useEffect, useState } from "react";

interface Props {
  text: string;
  active: boolean;
  speed?: number;
  delay?: number;
}

export function TypingText({ text, active, speed = 30, delay = 0 }: Props) {
  const [charCount, setCharCount] = useState(0);

  useEffect(() => {
    if (!active) {
      setCharCount(0);
      return;
    }

    setCharCount(0);

    const timeout = setTimeout(() => {
      const interval = setInterval(() => {
        setCharCount((prev) => {
          if (prev >= text.length) {
            clearInterval(interval);
            return prev;
          }
          return prev + 1;
        });
      }, speed);

      cleanupRef = () => clearInterval(interval);
    }, delay);

    let cleanupRef = () => {};

    return () => {
      clearTimeout(timeout);
      cleanupRef();
    };
  }, [active, text, speed, delay]);

  if (!active) return null;

  return (
    <>
      {text.slice(0, charCount)}
      {charCount < text.length && (
        <span className="inline-block h-4 w-[2px] animate-pulse bg-current" />
      )}
    </>
  );
}
