import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import type { Tone } from "@/components/styles/tones";
import {
  CheckmarkCircle02Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export const VISIBLE_CHATS = 6;

interface ChatStatus {
  icon: IconSvgElement;
  tone: Tone;
  spin: boolean;
  label: string;
}

const WORKING: ChatStatus = {
  icon: Loading03Icon,
  tone: "info",
  spin: true,
  label: "Working",
};
const DONE: ChatStatus = {
  icon: CheckmarkCircle02Icon,
  tone: "success",
  spin: false,
  label: "Done",
};

export function chatStatus(session: SessionSummaryResponse): ChatStatus {
  if (session.is_processing || session.chat_status === "queued") {
    return WORKING;
  }
  return DONE;
}

/** The recipient's latest chats; a null expert is Otto. */
export function recentChatsFor(
  sessions: SessionSummaryResponse[],
  expertID: string | null,
) {
  return sessions
    .filter((session) => (session.expert_id ?? null) === expertID)
    .slice(0, VISIBLE_CHATS);
}
