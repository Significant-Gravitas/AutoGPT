import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";

// Copilot action-gate approvals bind exact arguments; auto-approve cannot apply.
export const COPILOT_GATE_NODE_PREFIX = "copilot-node-gate-";

export function isGateReview(nodeId: string | null | undefined) {
  return !!nodeId?.startsWith(COPILOT_GATE_NODE_PREFIX);
}

// A chat's reviews wait on the expert whose chat raised them, or on Otto; only
// a workflow run is paused by its review.
export function waitingOnLine(groups: [string, PendingHumanReviewModel[]][]) {
  const inChat = groups.every(
    ([nodeId, nodeReviews]) =>
      isGateReview(nodeId) || nodeReviews.every((r) => !r.graph_exec_id),
  );
  if (!inChat)
    return "This workflow is paused until you approve the step below. Check what it will do, and edit it if needed.";
  const experts = new Set(
    groups.flatMap(([, nodeReviews]) => nodeReviews.map((r) => r.expert_name)),
  );
  const [expert] = experts;
  const name = experts.size === 1 && expert ? expert : AUTOPILOT_NAME;
  return `${name} is waiting for your approval before the action below.`;
}
