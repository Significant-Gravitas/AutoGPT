export function formatCents(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

export function formatMicrodollarsAsUsd(microdollars: number): string {
  const dollars = microdollars / 1_000_000;
  if (microdollars > 0 && dollars < 0.01) return "<$0.01";
  return `$${dollars.toFixed(2)}`;
}

export function formatResetTime(
  resetsAt: Date | string,
  now: Date = new Date(),
): string {
  const resetDate =
    typeof resetsAt === "string" ? new Date(resetsAt) : resetsAt;
  const diffMs = resetDate.getTime() - now.getTime();
  if (diffMs <= 0) return "now";

  const hours = Math.floor(diffMs / (1000 * 60 * 60));

  if (hours < 24) {
    const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    if (hours > 0) return `in ${hours}h ${minutes}m`;
    return `in ${minutes}m`;
  }

  return resetDate.toLocaleString(undefined, {
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
}
