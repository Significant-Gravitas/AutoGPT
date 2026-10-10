import { useRef, useState, type PointerEvent } from "react";

export function swipeChoice(dx: number, dy: number, width: number) {
  if (
    Math.abs(dx) < Math.max(64, Math.min(110, width * 0.25)) ||
    Math.abs(dx) < Math.abs(dy) * 1.5
  )
    return null;
  return dx < 0 ? 0 : 1;
}

export function useComparisonSwipe(
  disabled: boolean,
  choose: (index: number) => void,
) {
  const origin = useRef<{
    x: number;
    y: number;
    id: number;
    vertical: boolean;
  } | null>(null);
  const [offset, setOffset] = useState(0);
  const [candidate, setCandidate] = useState<number | null>(null);
  function cancel() {
    origin.current = null;
    setOffset(0);
    setCandidate(null);
  }
  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (
      disabled ||
      !event.isPrimary ||
      event.button !== 0 ||
      (event.pointerType === "mouse" &&
        window.matchMedia("(min-width: 768px)").matches)
    )
      return;
    if ((event.target as Element).closest("button, a, input, summary")) return;
    if (event.pointerType === "mouse") event.preventDefault();
    origin.current = {
      x: event.clientX,
      y: event.clientY,
      id: event.pointerId,
      vertical: false,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }
  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    const start = origin.current;
    if (!start || start.id !== event.pointerId) return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (Math.abs(dy) > 12 && Math.abs(dy) > Math.abs(dx)) start.vertical = true;
    if (start.vertical) return;
    setOffset(Math.max(-130, Math.min(130, dx)));
    setCandidate(swipeChoice(dx, dy, event.currentTarget.clientWidth));
  }
  function onPointerUp(event: PointerEvent<HTMLDivElement>) {
    const start = origin.current;
    if (!start || start.id !== event.pointerId) return;
    const index = swipeChoice(
      event.clientX - start.x,
      event.clientY - start.y,
      event.currentTarget.clientWidth,
    );
    if (!disabled && !start.vertical && index !== null) choose(index);
    cancel();
  }
  return {
    offset,
    candidate,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel: cancel,
    onLostPointerCapture: cancel,
  };
}
