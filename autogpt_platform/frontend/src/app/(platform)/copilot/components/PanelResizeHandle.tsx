"use client";

import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import { PANEL_RESERVED_WIDTH } from "../store";

interface Props {
  /** CSS selector for the panel element this handle resizes (e.g. "[data-artifact-panel]"). */
  panelSelector: string;
  onWidthChange: (width: number) => void;
  minWidth: number;
  maxWidth?: number;
  /** Space reserved for everything else in the flex row (chat + opposite rail). */
  reservedWidth?: number;
  /** Lets an animated panel drop its width transition while the user drags —
   *  otherwise every pointer move would queue a 300ms tween and the handle
   *  would trail the cursor. */
  onResizingChange?: (isResizing: boolean) => void;
}

// A drag reports at most one width per animation frame: pointer moves arrive
// faster than the screen paints, and every width written re-renders the panel.
interface PendingWidth {
  width: number | null;
  frame: number | null;
}

// Reports the pending width, if any, and cancels its scheduled frame. Called
// once per frame during a drag, and on pointer up/cancel/unmount before the
// drag ends, so the panel applies the last width with its tween still off.
function commitPendingWidth(
  pending: PendingWidth,
  onWidthChange: (width: number) => void,
) {
  if (pending.frame != null) {
    cancelAnimationFrame(pending.frame);
    pending.frame = null;
  }
  const width = pending.width;
  pending.width = null;
  if (width != null) onWidthChange(width);
}

export function PanelResizeHandle({
  panelSelector,
  onWidthChange,
  minWidth,
  maxWidth,
  reservedWidth = PANEL_RESERVED_WIDTH,
  onResizingChange,
}: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);
  const containerWidthRef = useRef(0);
  const onWidthChangeRef = useRef(onWidthChange);
  const minWidthRef = useRef(minWidth);
  const maxWidthRef = useRef(maxWidth);
  const reservedWidthRef = useRef(reservedWidth);
  onWidthChangeRef.current = onWidthChange;
  minWidthRef.current = minWidth;
  maxWidthRef.current = maxWidth;
  reservedWidthRef.current = reservedWidth;

  const pointerIdRef = useRef<number | null>(null);
  const onResizingChangeRef = useRef(onResizingChange);
  onResizingChangeRef.current = onResizingChange;

  const pendingRef = useRef<PendingWidth>({ width: null, frame: null });

  useEffect(() => {
    onResizingChangeRef.current?.(isDragging);
  }, [isDragging]);
  // A panel closed mid-drag unmounts the handle before the pointer comes up —
  // say the drag ended, or the panel opens at zero duration the next time.
  useEffect(() => () => onResizingChangeRef.current?.(false), []);

  // DOM integration: document-level pointer listeners bound only while
  // dragging, torn down on unmount so closing the panel mid-drag is safe.
  useEffect(() => {
    if (!isDragging) return;
    const pending = pendingRef.current;

    function handlePointerMove(moveEvent: PointerEvent) {
      const delta = startXRef.current - moveEvent.clientX;
      const available = containerWidthRef.current - reservedWidthRef.current;
      const cappedMax = maxWidthRef.current
        ? Math.min(maxWidthRef.current, available)
        : available;
      const effectiveMax = Math.max(minWidthRef.current, cappedMax);
      const newWidth = Math.min(
        effectiveMax,
        Math.max(minWidthRef.current, startWidthRef.current + delta),
      );
      pending.width = newWidth;
      if (pending.frame != null) return;
      pending.frame = requestAnimationFrame(() => {
        pending.frame = null;
        commitPendingWidth(pending, onWidthChangeRef.current);
      });
    }

    function handlePointerUp() {
      commitPendingWidth(pending, onWidthChangeRef.current);
      setIsDragging(false);
    }

    document.addEventListener("pointermove", handlePointerMove);
    document.addEventListener("pointerup", handlePointerUp);
    document.addEventListener("pointercancel", handlePointerUp);
    return () => {
      document.removeEventListener("pointermove", handlePointerMove);
      document.removeEventListener("pointerup", handlePointerUp);
      document.removeEventListener("pointercancel", handlePointerUp);
      // Closing the panel mid-drag still keeps the last width.
      commitPendingWidth(pending, onWidthChangeRef.current);
    };
  }, [isDragging]);

  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    e.preventDefault();
    startXRef.current = e.clientX;
    const panel = (e.target as HTMLElement).closest(
      panelSelector,
    ) as HTMLElement | null;
    startWidthRef.current = panel?.offsetWidth ?? minWidthRef.current;
    containerWidthRef.current =
      panel?.parentElement?.offsetWidth ??
      (typeof window !== "undefined"
        ? window.innerWidth
        : startWidthRef.current);
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
      pointerIdRef.current = e.pointerId;
    } catch {
      // Non-supporting environments (older test DOMs) — safe to ignore.
    }
    setIsDragging(true);
  }

  function handlePointerUp(e: React.PointerEvent<HTMLDivElement>) {
    if (pointerIdRef.current != null) {
      try {
        e.currentTarget.releasePointerCapture(pointerIdRef.current);
      } catch {
        // Capture may already be released.
      }
      pointerIdRef.current = null;
    }
    commitPendingWidth(pendingRef.current, onWidthChangeRef.current);
    setIsDragging(false);
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize panel"
      className="group absolute left-0 top-0 z-10 flex h-full w-3 -translate-x-1/2 cursor-col-resize items-stretch justify-center"
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      style={{ touchAction: "none" }}
    >
      <div
        className={cn(
          "h-full w-px bg-transparent transition-colors group-hover:w-0.5 group-hover:bg-violet-400",
          isDragging && "w-0.5 bg-violet-500",
        )}
      />
    </div>
  );
}
