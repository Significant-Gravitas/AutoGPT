"use client";

import { cn } from "@/lib/utils";
import { useCallback, useRef, useState } from "react";

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

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      setIsDragging(true);
      startXRef.current = e.clientX;

      // Get the panel's current width from its parent
      const panel = (e.target as HTMLElement).closest(
        "[data-artifact-panel]",
      ) as HTMLElement | null;
      startWidthRef.current = panel?.offsetWidth ?? 600;

      const handlePointerMove = (moveEvent: PointerEvent) => {
        const delta = startXRef.current - moveEvent.clientX;
        const maxWidth = window.innerWidth * (maxWidthPercent / 100);
        const newWidth = Math.min(
          maxWidth,
          Math.max(minWidth, startWidthRef.current + delta),
        );
        onWidthChange(newWidth);
      };

      const handlePointerUp = () => {
        setIsDragging(false);
        document.removeEventListener("pointermove", handlePointerMove);
        document.removeEventListener("pointerup", handlePointerUp);
      };

      document.addEventListener("pointermove", handlePointerMove);
      document.addEventListener("pointerup", handlePointerUp);
    },
    [onWidthChange, minWidth, maxWidthPercent],
  );

  return (
    <div
      className={cn(
        "absolute left-0 top-0 z-10 h-full w-1 cursor-col-resize transition-colors hover:w-1.5 hover:bg-violet-400",
        isDragging && "w-1.5 bg-violet-500",
      )}
      onPointerDown={handlePointerDown}
    />
  );
}
