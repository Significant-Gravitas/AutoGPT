import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";

const COPILOT_SESSION_PREFIX = "copilot-session-";

export function getReviewLink(review: PendingHumanReviewModel): string {
  if (review.session_id) return `/copilot?sessionId=${review.session_id}`;
  if (review.graph_exec_id.startsWith(COPILOT_SESSION_PREFIX)) {
    return `/copilot?sessionId=${review.graph_exec_id.slice(COPILOT_SESSION_PREFIX.length)}`;
  }
  if (review.library_agent_id) {
    return `/library/agents/${review.library_agent_id}?executionId=${review.graph_exec_id}`;
  }
  return "/library";
}

export function getReviewTitle(review: PendingHumanReviewModel): string {
  return review.instructions || review.agent_name || "Review needed";
}
