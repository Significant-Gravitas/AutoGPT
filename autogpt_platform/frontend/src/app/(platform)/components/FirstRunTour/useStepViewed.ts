"use client";

import { useStableCallback } from "@/hooks/useStableCallback";
import { useEffect, useRef } from "react";
import type { MeasuredStop } from "./useTourStops";

// How long a step's target must hold still before it is reported.
const STEP_SETTLE_MS = 500;

// Reports each step once, with what it points at once that has settled: on a
// phone the first measurement runs before the sidebar sheet opens, and would
// report the menu button instead of the item. A step left before it settles
// is still reported, with what it pointed at last.
export function useStepViewed(
  stop: MeasuredStop | undefined,
  report: (stop: MeasuredStop) => void,
) {
  const reported = useRef(new Set<string>());
  const latest = useRef(stop);
  useEffect(() => {
    latest.current = stop;
  });
  const reportOnce = useStableCallback(() => {
    const current = latest.current;
    if (!current || reported.current.has(current.stepID)) return;
    reported.current.add(current.stepID);
    report(current);
  });

  const stepID = stop?.stepID;
  const target = stop?.target;
  useEffect(() => {
    if (!stepID) return;
    const timer = window.setTimeout(reportOnce, STEP_SETTLE_MS);
    return () => window.clearTimeout(timer);
  }, [stepID, target, reportOnce]);
  useEffect(() => {
    if (stepID) return reportOnce;
  }, [stepID, reportOnce]);
}
