import { formatDistanceToNow } from "date-fns";

export function maskAPIKey(head: string, tail: string): string {
  return `${head}••••••••${tail}`;
}

export function formatLastUsed(lastUsedAt: Date | null | undefined): string {
  if (!lastUsedAt) return "Never used";
  return `Used ${formatDistanceToNow(lastUsedAt, { addSuffix: true })}`;
}
