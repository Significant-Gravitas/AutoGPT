import type { TeamProposal } from "../helpers";
import { str } from "../resultHelpers";

export function proposalName(proposal: TeamProposal): string {
  return str(proposal.preview, "name") ?? "New expert";
}

export function joinNames(names: string[]): string {
  if (names.length < 2) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** Cards cannot call tools, so confirming is a message that asks the model
 *  to make the call. One id uses the single-id parameter and several use the
 *  batch one — the tool rejects both at once. */
export function buildConfirmMessage(proposals: TeamProposal[]): string {
  const names = joinNames(proposals.map(proposalName));
  if (proposals.length === 1) {
    return `I approve ${names}. Call confirm_expert_change with confirmation_id "${proposals[0].confirmationId}".`;
  }
  const ids = proposals
    .map((proposal) => `"${proposal.confirmationId}"`)
    .join(", ");
  return `I approve ${names}. Call confirm_expert_change once with confirmation_ids [${ids}].`;
}
