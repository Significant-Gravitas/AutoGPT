"use client";

import { type RelativeTimeVariant, useRelativeTime } from "./useRelativeTime";

interface Props {
  date: Date | string | number | null | undefined;
  variant?: RelativeTimeVariant;
}

/**
 * `date` relative to now, e.g. "5 minutes ago" ("5m ago" for the short
 * variant), and nothing when there is no date. Safe in a server-rendered page:
 * the server's HTML carries the date itself, and the relative time follows
 * once the page hydrates (see useRelativeTime). It re-renders on the shared
 * minute clock, so the label keeps up with time without a timer of its own
 * and without its parent re-rendering.
 */
export function RelativeTime({ date, variant }: Props) {
  const label = useRelativeTime(date, variant);
  return <>{label}</>;
}
