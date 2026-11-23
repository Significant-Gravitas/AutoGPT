import { useEffect, useState } from "react";

/** How long "Copied" or "Saved" stays before the button reads as before. */
const FLASH_MS = 2_000;

/** A word a button wears for a moment after it acted ("Copied"). */
export function useFlash() {
  const [flashed, setFlashed] = useState<string | null>(null);
  useEffect(() => {
    if (!flashed) return;
    const timer = window.setTimeout(() => setFlashed(null), FLASH_MS);
    return () => window.clearTimeout(timer);
  }, [flashed]);
  return { flashed, flash: setFlashed };
}
