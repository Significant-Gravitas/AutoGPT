"use client";

import { useNowMinute } from "@/hooks/useNowMinute";
import { formatDistanceToNow } from "date-fns";

interface Props {
  date: Date | string | number;
}

/**
 * `date` relative to now, e.g. "5 minutes ago". It re-renders on the shared
 * minute clock, so the label keeps up with time without a timer of its own
 * and without its parent re-rendering.
 */
export function RelativeTime({ date }: Props) {
  useNowMinute();
  return <>{formatDistanceToNow(date, { addSuffix: true })}</>;
}
