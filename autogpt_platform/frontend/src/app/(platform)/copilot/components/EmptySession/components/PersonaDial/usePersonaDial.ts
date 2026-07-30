"use client";

import { animate, useMotionValue } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import {
  angleFromCentre,
  clampRotation,
  indexFromRotation,
  nearestVirtual,
  rotationForVirtual,
  shouldWrap,
  snapRotation,
  virtualFromRotation,
  wrapIndex,
} from "./helpers";
import { playDialTick } from "./tickSound";

// Spring (not duration) so releases and snaps carry momentum and settle
// naturally; tuned stiff enough to stay under ~400ms.
const SPRING = { type: "spring", stiffness: 220, damping: 30 } as const;

interface Args {
  count: number;
  selectedIndex: number;
  onSelect: (index: number) => void;
  /** Fired when the pointer is released without having dragged (a tap). */
  onTap?: (point: { x: number; y: number }) => void;
}

// Rotation lives in a MotionValue: drags and springs update the DOM directly
// without re-rendering React per frame. React only re-renders when the wheel
// crosses a slot boundary (to shift the virtual window and live-select).
export function usePersonaDial({
  count,
  selectedIndex,
  onSelect,
  onTap,
}: Args) {
  const wrap = shouldWrap(count);
  const ringRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{
    angle: number;
    rotation: number;
    moved: boolean;
  } | null>(null);
  const rotation = useMotionValue(rotationForVirtual(selectedIndex));
  const lastIndexRef = useRef(selectedIndex);
  const lastVirtualRef = useRef(selectedIndex);
  const [virtualCentre, setVirtualCentre] = useState(selectedIndex);
  const [isDragging, setIsDragging] = useState(false);

  // Live-select: whichever persona crosses the bottom point becomes selected,
  // during drags and mid-spring alike. Each slot crossing plays a ratchet tick.
  useEffect(() => {
    return rotation.on("change", (value) => {
      const virtual = virtualFromRotation(value);
      if (virtual !== lastVirtualRef.current) {
        lastVirtualRef.current = virtual;
        playDialTick();
      }
      setVirtualCentre(virtual);
      const index = indexFromRotation(value, count);
      if (index !== lastIndexRef.current) {
        lastIndexRef.current = index;
        onSelect(index);
      }
    });
  });

  // Selection changed from outside (search arrows, Enter-to-pick): spring the
  // wheel round to it by the shortest path.
  useEffect(() => {
    if (selectedIndex !== lastIndexRef.current) {
      lastIndexRef.current = selectedIndex;
      animate(
        rotation,
        rotationForVirtual(
          wrap
            ? nearestVirtual(selectedIndex, count, rotation.get())
            : selectedIndex,
        ),
        SPRING,
      );
    }
  }, [selectedIndex, count, rotation]);

  function centreOf() {
    const box = ringRef.current?.getBoundingClientRect();
    if (!box) return null;
    return { x: box.left + box.width / 2, y: box.top + box.height / 2 };
  }

  function handlePointerDown(e: ReactPointerEvent<HTMLDivElement>) {
    const centre = centreOf();
    if (!centre) return;
    rotation.stop();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // Capture can fail for exotic/synthetic pointers; drag still works,
      // it just won't survive the pointer leaving the ring.
    }
    dragRef.current = {
      angle: angleFromCentre(centre, { x: e.clientX, y: e.clientY }),
      rotation: rotation.get(),
      moved: false,
    };
    setIsDragging(true);
  }

  function handlePointerMove(e: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    const centre = centreOf();
    if (!drag || !centre) return;
    const angle = angleFromCentre(centre, { x: e.clientX, y: e.clientY });
    let delta = angle - drag.angle;
    if (delta > 180) delta -= 360;
    if (delta < -180) delta += 360;
    const raw = drag.rotation + delta;
    const next = wrap ? raw : clampRotation(raw, count);
    if (Math.abs(next - drag.rotation) > 2) drag.moved = true;
    rotation.set(next);
  }

  function handlePointerUp(e: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // Mirror of setPointerCapture above — nothing to release.
    }
    dragRef.current = null;
    setIsDragging(false);
    if (!drag.moved) {
      // Pointer capture swallows the click event on rim items, so taps have
      // to be resolved here from the release point.
      onTap?.({ x: e.clientX, y: e.clientY });
      return;
    }
    const snapped = snapRotation(rotation.get());
    animate(rotation, wrap ? snapped : clampRotation(snapped, count), SPRING);
  }

  function selectVirtual(virtual: number) {
    const target = wrap ? virtual : Math.min(count - 1, Math.max(0, virtual));
    lastIndexRef.current = wrapIndex(target, count);
    onSelect(lastIndexRef.current);
    animate(rotation, rotationForVirtual(target), SPRING);
  }

  function step(direction: 1 | -1) {
    selectVirtual(virtualFromRotation(rotation.get()) + direction);
  }

  return {
    ringRef,
    rotation,
    virtualCentre,
    isDragging,
    selectVirtual,
    step,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
  };
}
