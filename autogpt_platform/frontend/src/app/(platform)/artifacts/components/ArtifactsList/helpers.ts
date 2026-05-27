export type FileOrigin =
  | { kind: "session"; sessionId: string; href: string }
  | { kind: "builder"; href: string };

const SESSION_PATH_RE = /^\/sessions\/([^/]+)\//;

export function deriveFileOrigin(filePath: string | undefined): FileOrigin {
  const match = (filePath ?? "").match(SESSION_PATH_RE);
  if (match) {
    const sessionId = match[1];
    return {
      kind: "session",
      sessionId,
      href: `/copilot?sessionId=${encodeURIComponent(sessionId)}`,
    };
  }
  return { kind: "builder", href: "/build" };
}

export function getFileDownloadUrl(fileId: string): string {
  return `/api/proxy/api/workspace/files/${encodeURIComponent(fileId)}/download`;
}

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const exp = Math.min(
    BYTE_UNITS.length - 1,
    Math.floor(Math.log(bytes) / Math.log(1024)),
  );
  const value = bytes / Math.pow(1024, exp);
  const formatted = exp === 0 ? value.toFixed(0) : value.toFixed(1);
  return `${formatted} ${BYTE_UNITS[exp]}`;
}

export function formatRelativeDate(input: string | Date): string {
  const date = input instanceof Date ? input : new Date(input);
  if (Number.isNaN(date.getTime())) return "—";

  const diffMs = Date.now() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;

  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: date.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
  });
}
