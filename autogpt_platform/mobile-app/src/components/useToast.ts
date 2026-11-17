import { useRef, useState } from "react";
import { AccessibilityInfo } from "react-native";

import { useMountEffect } from "@/lib/useMountEffect";

const TOAST_MS = 4_000;

export interface ToastState {
  message: string;
  tone: "success" | "error";
  actionLabel?: string;
  onAction?: () => void;
}

/** One toast at a time; a new one replaces the last and each hides by itself. */
export function useToast() {
  const [toast, setToast] = useState<ToastState | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  function clearTimer() {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  }

  function show(next: ToastState) {
    clearTimer();
    setToast(next);
    // iOS reads no live region out loud; announce it instead.
    AccessibilityInfo.announceForAccessibility(next.message);
    timer.current = setTimeout(hide, TOAST_MS);
  }

  function hide() {
    clearTimer();
    setToast(null);
  }

  useMountEffect(() => clearTimer);

  return { toast, show, hide };
}
