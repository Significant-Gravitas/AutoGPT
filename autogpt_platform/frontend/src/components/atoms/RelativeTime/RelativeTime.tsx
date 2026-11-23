"use client";

import { useEffect, useState } from "react";
import { useNowMinute } from "@/hooks/useNowMinute";
import { formatDistanceToNow } from "date-fns";

interface Props {
  date: Date | string | number;
}

export function RelativeTime({ date }: Props) {
  useNowMinute();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const parsed = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(parsed.getTime())) {
    return <>{String(date)}</>;
  }

  if (!mounted) {
    return <>{parsed.toISOString().slice(0, 10)}</>;
  }

  return <>{formatDistanceToNow(parsed, { addSuffix: true })}</>;
}
