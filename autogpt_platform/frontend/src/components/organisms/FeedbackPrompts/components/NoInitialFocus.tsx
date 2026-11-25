"use client";

import { useEffect, useRef } from "react";

export function NoInitialFocus() {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const frame = requestAnimationFrame(() => ref.current?.focus());
    return () => cancelAnimationFrame(frame);
  }, []);

  return <span ref={ref} tabIndex={-1} className="sr-only" />;
}
