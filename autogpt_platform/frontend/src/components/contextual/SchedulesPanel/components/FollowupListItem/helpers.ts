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

  // ``session_id`` is null when the follow-up is the "fresh chat at fire
  // time" sentinel — there is no destination session to link to until the
  // schedule fires.  Surface that explicitly to the renderer so it can
  // disable the link and show a "New chat" pill instead.
  const sessionHref = followup.session_id
    ? `/copilot?sessionId=${encodeURIComponent(followup.session_id)}`
    : null;

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
