"use client";

import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import { DEFAULT_PANEL_WIDTH } from "../../../store";

interface Props {
  onWidthChange: (width: number) => void;
  minWidth?: number;
  maxWidthPercent?: number;
}

export function ArtifactDragHandle({
  onWidthChange,
  minWidth = 320,
  maxWidthPercent = 85,
}: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);
  // Use refs for the callback + bounds so the drag listeners can read the
  // latest values without having to detach/reattach between re-renders.
  const onWidthChangeRef = useRef(onWidthChange);
  const minWidthRef = useRef(minWidth);
  const maxWidthPercentRef = useRef(maxWidthPercent);
  onWidthChangeRef.current = onWidthChange;
  minWidthRef.current = minWidth;
  maxWidthPercentRef.current = maxWidthPercent;

  // Track the captured pointer id so pointerup can release it even after
  // React re-renders.
  const pointerIdRef = useRef<number | null>(null);

  // Attach document listeners only while dragging, and always tear them down
  // on unmount — otherwise closing the panel mid-drag leaves listeners bound
  // to a handler that calls setState on the unmounted component.
  useEffect(() => {
    if (!isDragging) return;

    function handlePointerMove(moveEvent: PointerEvent) {
      const delta = startXRef.current - moveEvent.clientX;
      const maxWidth = window.innerWidth * (maxWidthPercentRef.current / 100);
      const newWidth = Math.min(
        maxWidth,
        Math.max(minWidthRef.current, startWidthRef.current + delta),
      );
      onWidthChangeRef.current(newWidth);
    }

    function handlePointerUp() {
      setIsDragging(false);
    }

    document.addEventListener("pointermove", handlePointerMove);
    document.addEventListener("pointerup", handlePointerUp);
    document.addEventListener("pointercancel", handlePointerUp);
    return () => {
      document.removeEventListener("pointermove", handlePointerMove);
      document.removeEventListener("pointerup", handlePointerUp);
      document.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [isDragging]);

  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    e.preventDefault();
    startXRef.current = e.clientX;

    // Get the panel's current width from its parent
    const panel = (e.target as HTMLElement).closest(
      "[data-artifact-panel]",
    ) as HTMLElement | null;
    startWidthRef.current = panel?.offsetWidth ?? DEFAULT_PANEL_WIDTH;

    // Capture the pointer so pointermove/pointerup still reach us when the
    // cursor drifts over sandboxed artifact iframes. Without this, the iframe
    // eats the events and the drag gets stuck (SECRT-2256).
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
    setIsDragging(false);
  }

  return (
    // 12px transparent hit target with the visible 1px line centered inside
    // (WCAG-compliant, matches ~8-12px conventions of other resizable panels).
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize panel"
      className={cn(
        "group absolute -left-1.5 top-0 z-10 flex h-full w-3 cursor-col-resize items-stretch justify-center",
      )}
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
