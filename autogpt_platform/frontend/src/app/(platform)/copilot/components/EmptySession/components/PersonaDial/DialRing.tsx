"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useEffect, useRef } from "react";
import type { Persona } from "../../personas";
import { DialItem } from "./DialItem";
import {
  DIAL_ITEM_RADIUS,
  DIAL_RADIUS,
  DIAL_STEP,
  DIAL_WINDOW,
  shouldWrap,
  wrapIndex,
} from "./helpers";
import { usePersonaDial } from "./usePersonaDial";

interface Props {
  personas: Persona[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  onClose: () => void;
}

// The drag surface reaches the OUTER edge of the rim avatars, not just their
// centres, so grabbing anywhere on an avatar (or between them) spins the dial.
const SURFACE = DIAL_RADIUS + DIAL_ITEM_RADIUS;

export function DialRing({
  personas,
  selectedIndex,
  onSelect,
  onClose,
}: Props) {
  const count = personas.length;
  // Selection mode stays open until the user picks a persona or clicks outside
  // the picker. Taps are resolved via onTap because the ring's pointer capture
  // swallows click events on items.
  const dial = usePersonaDial({
    count,
    selectedIndex,
    onSelect,
    onTap: handleTap,
  });
  const closeTimer = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (closeTimer.current) window.clearTimeout(closeTimer.current);
    };
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      const inInput = (e.target as HTMLElement | null)?.tagName === "INPUT";
      if (inInput) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        dial.step(-1);
      }
      if (e.key === "ArrowRight") {
        e.preventDefault();
        dial.step(1);
      }
    }
    function onPointerDownOutside(e: PointerEvent) {
      const target = e.target as Element | null;
      if (!target?.closest("[data-persona-picker]")) onClose();
    }
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onPointerDownOutside);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onPointerDownOutside);
    };
  });

  function handleItemPick(virtual: number) {
    // Let the wheel visibly scroll the pick down to the avatar before leaving
    // selection mode.
    dial.selectVirtual(virtual);
    if (closeTimer.current) window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(onClose, 650);
  }

  function handleTap(point: { x: number; y: number }) {
    // Resolved geometrically (angle around the ring centre) rather than via
    // elementFromPoint, which overlapping banners/toasts can intercept.
    const box = dial.ringRef.current?.getBoundingClientRect();
    if (!box) return;
    const dx = point.x - (box.left + box.width / 2);
    const dy = point.y - (box.top + box.height / 2);
    if (Math.abs(Math.hypot(dx, dy) - DIAL_RADIUS) > DIAL_ITEM_RADIUS) return;
    const theta = (Math.atan2(-dx, dy) * 180) / Math.PI;
    const virtual = Math.round((theta - dial.rotation.get()) / DIAL_STEP);
    if (!shouldWrap(count) && (virtual < 0 || virtual >= count)) return;
    handleItemPick(virtual);
  }

  // Large rosters wrap forever (virtual slots around the current centre);
  // small ones — like filtered search results — show each persona exactly
  // once on a bounded arc, so no duplicates appear.
  const slots = shouldWrap(count)
    ? Array.from(
        { length: DIAL_WINDOW * 2 + 1 },
        (_, i) => dial.virtualCentre - DIAL_WINDOW + i,
      )
    : Array.from({ length: count }, (_, i) => i);

  return (
    <motion.div
      ref={dial.ringRef}
      onPointerDown={dial.handlePointerDown}
      onPointerMove={dial.handlePointerMove}
      onPointerUp={dial.handlePointerUp}
      onPointerCancel={dial.handlePointerUp}
      className={cn(
        // `absolute` is load-bearing: the rim items are absolutely positioned
        // and must resolve against the ring. Without it they'd fall back to
        // the page whenever framer renders rotate(0) as no transform at all.
        "absolute rounded-full",
        "touch-none select-none",
        dial.isDragging ? "cursor-grabbing" : "cursor-grab",
      )}
      style={{
        width: SURFACE * 2,
        height: SURFACE * 2,
        left: -SURFACE,
        top: -(DIAL_RADIUS + SURFACE),
        rotate: dial.rotation,
      }}
    >
      {slots.map((virtual) => (
        <DialItem
          key={virtual}
          persona={personas[wrapIndex(virtual, count)]}
          index={virtual}
          isSelected={wrapIndex(virtual, count) === selectedIndex}
          entranceDelay={
            0.05 + Math.abs(virtual - dial.virtualCentre) * 0.06
          }
          rotation={dial.rotation}
          onPick={handleItemPick}
        />
      ))}
    </motion.div>
  );
}
