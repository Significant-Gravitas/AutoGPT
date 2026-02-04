"use client";

import { useEffect, useState } from "react";

const TYPING_SPEED = 30;

interface Props {
  text: string;
  className?: string;
}

export function TypeWriter({ text, className }: Props) {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    setDisplayedText("");
    let currentIndex = 0;

    const interval = setInterval(() => {
      if (currentIndex < text.length) {
        setDisplayedText(text.slice(0, currentIndex + 1));
        currentIndex++;
      } else {
        clearInterval(interval);
      }
    }, TYPING_SPEED);

    return () => clearInterval(interval);
  }, [text]);

  return (
    <span className={className}>
      {displayedText}
      <span className="animate-pulse">|</span>
    </span>
  );
}
