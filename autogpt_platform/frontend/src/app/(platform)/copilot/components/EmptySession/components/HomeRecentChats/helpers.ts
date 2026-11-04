import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import type { Tone } from "@/components/styles/tones";
import {
  AlertCircleIcon,
  CheckmarkCircle02Icon,
  Loading03Icon,
  MultiplicationSignCircleIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

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
const NEEDS_YOU: ChatStatus = {
  icon: AlertCircleIcon,
  tone: "waiting",
  spin: false,
  label: "Needs you",
};
const FAILED: ChatStatus = {
  icon: MultiplicationSignCircleIcon,
  tone: "danger",
  spin: false,
  label: "Failed",
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
  if (session.attention === "needs_you") return NEEDS_YOU;
  if (session.attention === "failed") return FAILED;
  return DONE;
}
