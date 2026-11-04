"use client";

import { formatElapsed } from "../../helpers";
import { useNow } from "./useNow";

interface Props {
  since: string | null;
}

/** Seconds since `since`, ticking once a second. Wrap it in a tabular Text. */
export function LiveTimer({ since }: Props) {
  const now = useNow();
  return <>{formatElapsed(since, now)}</>;
}
