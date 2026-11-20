import type { FeedbackPromptState } from "@/app/api/__generated__/models/feedbackPromptState";
import type { FeedbackPromptToolResponse } from "@/app/api/__generated__/models/feedbackPromptToolResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { WorkCheckRating } from "@/components/organisms/FeedbackPrompts/WorkCheckCard";
import type { ToolUIPart } from "ai";

export const WORK_CHECK_PART_TYPE = "tool-request_work_check";
export const MISTAKE_PART_TYPE = "tool-flag_mistake";

/** The reply a card rates. Absent when the card must not appear. */
export interface FeedbackCardTarget {
  sessionID: string;
  messageID: string;
}

export function isFeedbackCardPart(part: { type: string }) {
  return part.type === WORK_CHECK_PART_TYPE || part.type === MISTAKE_PART_TYPE;
}

function toToolResponse(output: unknown): FeedbackPromptToolResponse | null {
  if (typeof output === "string") {
    try {
      return toToolResponse(JSON.parse(output));
    } catch {
      return null;
    }
  }
  if (!output || typeof output !== "object") return null;
  const response = output as FeedbackPromptToolResponse;
  return response.type === ResponseType.feedback_prompt ? response : null;
}

export function readFeedbackPrompt(part: ToolUIPart) {
  if (part.state !== "output-available") return null;
  const response = toToolResponse(part.output);
  if (!response?.prompt_id) return null;
  return { promptId: response.prompt_id, isFirst: Boolean(response.is_first) };
}

export function getWorkCheckNote(isFirst: boolean) {
  return isFirst
    ? "Quick check-in on your first finished task"
    : "Quick check-in on this task";
}

export const RATING_SCORES: Record<WorkCheckRating, 0 | 1 | 2> = {
  bad: 0,
  okay: 1,
  great: 2,
};

export function ratingFromState(
  state: FeedbackPromptState,
): WorkCheckRating | null {
  if (!state.answered) return null;
  if (state.score === 2) return "great";
  return state.score === 0 ? "bad" : "okay";
}
