import { useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState, type RefObject } from "react";

const BLINK_MS = 130;
const BLINK_MIN_GAP_MS = 2600;
const BLINK_MAX_GAP_MS = 5200;
const GAZE_LIMIT = 3;

interface Args {
  animated: boolean;
  trackPointer: boolean;
  svgRef: RefObject<SVGSVGElement | null>;
}

export function useBotAvatar({ animated, trackPointer, svgRef }: Args) {
  const reducedMotion = useReducedMotion();
  const isLive = animated && !reducedMotion;
  const [isBlinking, setIsBlinking] = useState(false);
  const [gaze, setGaze] = useState({ x: 0, y: 0 });
  const frame = useRef<number | null>(null);

  useEffect(() => {
    if (!isLive) return;
    let timer: ReturnType<typeof setTimeout>;
    function schedule() {
      const gap =
        BLINK_MIN_GAP_MS +
        Math.random() * (BLINK_MAX_GAP_MS - BLINK_MIN_GAP_MS);
      timer = setTimeout(() => {
        setIsBlinking(true);
        timer = setTimeout(() => {
          setIsBlinking(false);
          schedule();
        }, BLINK_MS);
      }, gap);
    }
    schedule();
    return () => clearTimeout(timer);
  }, [isLive]);

  useEffect(() => {
    if (!isLive || !trackPointer) return;
    function handleMove(event: PointerEvent) {
      if (frame.current !== null) return;
      frame.current = requestAnimationFrame(() => {
        frame.current = null;
        const box = svgRef.current?.getBoundingClientRect();
        if (!box) return;
        const dx = event.clientX - (box.left + box.width / 2);
        const dy = event.clientY - (box.top + box.height / 2);
        const distance = Math.hypot(dx, dy) || 1;
        const reach = Math.min(1, distance / 260) * GAZE_LIMIT;
        setGaze({ x: (dx / distance) * reach, y: (dy / distance) * reach });
      });
    }
    window.addEventListener("pointermove", handleMove);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      if (frame.current !== null) cancelAnimationFrame(frame.current);
    };
  }, [isLive, trackPointer, svgRef]);

  return { isLive, isBlinking, gaze };
}
