import type {
  ChatDelegation,
  LiveDelegationStatus,
} from "../../../../delegations";

/** How many hand-offs wait on the user: approvals plus questions. */
export function countWaiting(
  delegations: ChatDelegation[],
  liveStatuses: Record<string, LiveDelegationStatus>,
) {
  return delegations.filter((delegation) => {
    const status = liveStatuses[delegation.toolCallId] ?? delegation.status;
    return status === "proposed" || status === "needs-input";
  }).length;
}
