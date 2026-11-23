"use client";

import { useEffect, useState } from "react";
import { useNowMinute } from "@/hooks/useNowMinute";
import { formatDistanceToNow } from "date-fns";

interface Props {
  date: Date | string | number;
}

function formatUtcDate(date: Date | string | number): string {
  const parsed = date instanceof Date ? date : new Date(date);
  return Number.isNaN(parsed.getTime())
    ? String(date)
    : parsed.toISOString().slice(0, 10);
}

/**
 * `date` relative to now, e.g. "5 minutes ago". It re-renders on the shared
 * minute clock, so the label keeps up with time without a timer of its own
 * and without its parent re-rendering.
 *
 * Renders the deterministic UTC date until mount so SSR and client hydration
 * always match regardless of clock drift or elapsed time.
 */
export function RelativeTime({ date }: Props) {
  useNowMinute();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <>{formatUtcDate(date)}</>;
  }

  return <>{formatDistanceToNow(date, { addSuffix: true })}</>;
}
