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

export function formatTierLabel(
  tier: string | null | undefined,
): string | null {
  if (!tier || tier === "NO_TIER") return null;
  return tier.charAt(0).toUpperCase() + tier.slice(1).toLowerCase();
}

export const TIER_BADGE_CLASS_NAME = "bg-[rgb(224,237,255)]";

interface UsageWindowLike {
  percent_used?: number | null;
}

interface UsageLike {
  daily?: UsageWindowLike | null;
  weekly?: UsageWindowLike | null;
}

export function isUsageExhausted(usage: UsageLike | null | undefined): boolean {
  if (!usage) return false;
  const daily = usage.daily?.percent_used ?? 0;
  const weekly = usage.weekly?.percent_used ?? 0;
  return daily >= 100 || weekly >= 100;
}

export function formatBytes(bytes: number): string {
  const KB = 1024;
  const MB = KB * 1024;
  const GB = MB * 1024;
  if (bytes < KB) return `${bytes} B`;
  if (bytes < MB) {
    const kb = Math.round(bytes / KB);
    return kb >= 1024 ? `${(bytes / MB).toFixed(1)} MB` : `${kb} KB`;
  }
  if (bytes < GB) {
    const mb = Math.round(bytes / MB);
    return mb >= 1024 ? `${(bytes / GB).toFixed(1)} GB` : `${mb} MB`;
  }
  return `${(bytes / GB).toFixed(1)} GB`;
}
