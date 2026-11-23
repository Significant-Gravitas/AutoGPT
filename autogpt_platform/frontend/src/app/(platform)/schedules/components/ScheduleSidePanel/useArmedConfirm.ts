import { isKey } from "@/lib/keyboard";
import { useEffect, useState } from "react";

/** How long an armed confirm waits for its second click. */
export const DISARM_MS = 5_000;

/** The schedule whose ending is armed: it asks once more before it acts.
 *  It disarms on Escape, when focus moves off its two buttons, when
 *  another schedule opens and after a few seconds, so nothing is left
 *  armed for a stray click later. */
export function useArmedConfirm(row: string | null, armedLabels: string[]) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const [openRow, setOpenRow] = useState(row);
  if (openRow !== row) {
    setOpenRow(row);
    setConfirming(null);
  }
  const labels = armedLabels.join("\n");

  useEffect(() => {
    if (!confirming) return;
    const kept = new Set(labels.split("\n"));
    const timer = window.setTimeout(() => setConfirming(null), DISARM_MS);
    function onKeyDown(e: KeyboardEvent) {
      if (isKey(e, "Escape")) setConfirming(null);
    }
    function onFocusIn(e: FocusEvent) {
      const target = e.target;
      const label =
        target instanceof HTMLButtonElement ? target.textContent?.trim() : "";
      if (!label || !kept.has(label)) setConfirming(null);
    }
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("focusin", onFocusIn);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("focusin", onFocusIn);
    };
  }, [confirming, labels]);

  return { confirming, setConfirming };
}
