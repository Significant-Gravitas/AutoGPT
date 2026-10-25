"use client";

import { useEffect, useState } from "react";
import {
  getSpotlightBox,
  isSameBox,
  resolveStop,
  type Box,
  type ResolvedStop,
} from "./helpers";
import type { TourStep } from "./steps";

export interface MeasuredStop extends ResolvedStop {
  /** The step this was measured for: a stale stop is never shown as current. */
  stepID: string;
  box: Box | null;
}

function isSameStop(a: MeasuredStop, b: MeasuredStop) {
  return (
    a.stepID === b.stepID &&
    a.element === b.element &&
    a.target === b.target &&
    a.viaMenu === b.viaMenu &&
    isSameBox(a.box, b.box)
  );
}

// Follows targets as the layout moves: the sidebar sliding, scroll, resize,
// and targets that mount late.
export function useTourStops(steps: TourStep[]) {
  const [stops, setStops] = useState<MeasuredStop[]>([]);

  useEffect(() => {
    if (steps.length === 0) return;

    let frame = 0;
    const observed = new Set<HTMLElement>();
    const resizeObserver = new ResizeObserver(schedule);

    function measure() {
      frame = 0;
      const next = steps.map((step) => {
        const stop = resolveStop(step);
        return {
          ...stop,
          stepID: step.id,
          box: stop.element ? getSpotlightBox(stop.element, stop.target) : null,
        };
      });
      const elements = new Set(
        next.flatMap((stop) => (stop.element ? [stop.element] : [])),
      );
      for (const element of Array.from(observed)) {
        if (elements.has(element)) continue;
        resizeObserver.unobserve(element);
        observed.delete(element);
      }
      for (const element of Array.from(elements)) {
        if (observed.has(element)) continue;
        resizeObserver.observe(element);
        observed.add(element);
      }
      setStops((previous) =>
        previous.length === next.length &&
        previous.every((stop, index) => isSameStop(stop, next[index]))
          ? previous
          : next,
      );
    }

    function schedule() {
      if (!frame) frame = requestAnimationFrame(measure);
    }

    measure();
    // Keep measuring through the sidebar's width transition.
    const settle = window.setInterval(schedule, 50);
    const stopSettling = window.setTimeout(
      () => window.clearInterval(settle),
      700,
    );
    const mutationObserver = new MutationObserver(schedule);
    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-state", "data-collapsible", "hidden"],
    });
    window.addEventListener("resize", schedule);
    window.addEventListener("scroll", schedule, true);

    return () => {
      if (frame) cancelAnimationFrame(frame);
      window.clearInterval(settle);
      window.clearTimeout(stopSettling);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
      window.removeEventListener("resize", schedule);
      window.removeEventListener("scroll", schedule, true);
    };
  }, [steps]);

  return stops;
}
