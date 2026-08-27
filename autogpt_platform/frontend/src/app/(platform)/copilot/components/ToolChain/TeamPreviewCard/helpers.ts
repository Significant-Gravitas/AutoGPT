import type { TeamProposal } from "../helpers";
import { str } from "../resultHelpers";

export function proposalName(proposal: TeamProposal): string {
  return str(proposal.preview, "name") ?? "New expert";
}

export function joinNames(names: string[]): string {
  if (names.length < 2) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function buildConfirmMessage(proposals: TeamProposal[]): string {
  const names = joinNames(proposals.map(proposalName));
  if (proposals.length === 1) {
    return `I approve ${names}. Add them to my team.`;
  }
  return `I approve ${names}. Add them to my team in one step.`;
}
