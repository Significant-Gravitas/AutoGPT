import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";

const MESSAGE_PREVIEW_MAX_LEN = 140;

export function describeFollowup(followup: CopilotTurnJobInfo) {
  const message = (followup.message ?? "").trim();
  const messagePreview =
    message.length === 0
      ? "(no message)"
      : message.length > MESSAGE_PREVIEW_MAX_LEN
        ? `${message.slice(0, MESSAGE_PREVIEW_MAX_LEN).trimEnd()}…`
        : message;

  const sessionHref = `/copilot?sessionId=${encodeURIComponent(followup.session_id)}`;

  return { messagePreview, sessionHref };
}

export function formatNextRunTitle(
  nextRunIso: string | null | undefined,
  scheduleTimezone: string | null | undefined,
): string | undefined {
  if (!nextRunIso) return undefined;
  const date = new Date(nextRunIso);
  if (Number.isNaN(date.valueOf())) return undefined;

  const tz =
    scheduleTimezone && scheduleTimezone.trim().length > 0
      ? scheduleTimezone
      : Intl.DateTimeFormat().resolvedOptions().timeZone;

  try {
    const formatted = new Intl.DateTimeFormat(undefined, {
      timeZone: tz,
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    }).format(date);
    return `Next run: ${formatted} (${tz})`;
  } catch {
    return `Next run: ${date.toISOString()} (UTC)`;
  }
}
