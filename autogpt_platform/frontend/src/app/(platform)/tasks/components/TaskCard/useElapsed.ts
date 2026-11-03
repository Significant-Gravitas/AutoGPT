import { useEffect, useState } from "react";

export function useElapsed(startedAt: string | null) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!startedAt) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [startedAt]);

  if (!startedAt) return null;
  const started = new Date(startedAt).getTime();
  if (Number.isNaN(started)) return null;
  return Math.max(0, now - started);
}
