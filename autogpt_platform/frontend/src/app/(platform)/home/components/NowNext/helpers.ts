const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;

export function formatUntil(nextRunTime: Date, now = new Date()) {
  const runAt = new Date(nextRunTime);
  const diffMs = runAt.getTime() - now.getTime();
  if (Number.isNaN(runAt.getTime())) return "";
  if (diffMs <= 60_000) return "now";
  if (diffMs < DAY_MS) {
    const totalMinutes = Math.round(diffMs / 60_000);
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    if (hours === 0) return `in ${minutes}m`;
    return minutes > 0 ? `in ${hours}h ${minutes}m` : `in ${hours}h`;
  }
  return runAt.toLocaleString(undefined, {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatRunningFor(
  startedAt: Date | null | undefined,
  now = new Date(),
) {
  if (!startedAt) return null;
  const started = new Date(startedAt);
  if (Number.isNaN(started.valueOf())) return null;
  const totalMinutes = Math.floor((now.getTime() - started.getTime()) / 60_000);
  if (totalMinutes < 1) return "Running for less than a minute";
  if (totalMinutes < 60) return `Running for ${totalMinutes}m`;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes > 0
    ? `Running for ${hours}h ${minutes}m`
    : `Running for ${hours}h`;
}
