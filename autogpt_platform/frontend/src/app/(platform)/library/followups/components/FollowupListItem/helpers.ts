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
