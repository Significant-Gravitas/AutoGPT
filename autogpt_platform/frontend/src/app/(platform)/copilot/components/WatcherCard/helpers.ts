export interface WatcherMetadata {
  title: string;
  description: string;
  actionLabel: string;
  actionHref: string;
  status: "blocked" | "failed";
}

function asText(value: unknown, fallback: string, maxLength: number) {
  return typeof value === "string" && value.trim()
    ? value.trim().slice(0, maxLength)
    : fallback;
}

function safeActionHref(value: unknown) {
  if (typeof value !== "string") return "/home";
  const allowed =
    value === "/home" ||
    value.startsWith("/team?") ||
    value.startsWith("/library/agents/");
  return allowed && !value.startsWith("//") ? value : "/home";
}

export function getWatcherMetadata(value: unknown): WatcherMetadata | null {
  if (!value || typeof value !== "object") return null;
  const metadata = value as Record<string, unknown>;
  if (metadata.kind !== "copilot_watcher") return null;
  return {
    title: asText(metadata.title, "Expert update needs attention", 120),
    description: asText(
      metadata.description,
      "Open Home for the current status.",
      240,
    ),
    actionLabel: asText(metadata.action_label, "Open Home", 40),
    actionHref: safeActionHref(metadata.action_href),
    status: metadata.status === "failed" ? "failed" : "blocked",
  };
}
