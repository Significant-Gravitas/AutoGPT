/**
 * Formats an ISO date string into a human-readable relative time string.
 * e.g. "3m ago", "2h ago", "5d ago".
 */
export function formatTimeAgo(isoDate: string): string {
  const parsed = new Date(isoDate).getTime();
  if (Number.isNaN(parsed)) return "unknown";
  const diff = Date.now() - parsed;
  if (diff < 0) return "just now";
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
