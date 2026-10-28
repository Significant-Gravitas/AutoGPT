import type { ExpertSetupItem } from "@/app/api/__generated__/models/expertSetupItem";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";

export function isCredentialItem(item: ExpertSetupItem) {
  return item.resolution === "connect" || item.resolution === "allow";
}

export function getSetupRowCopy(item: ExpertSetupItem) {
  const workflowName = item.workflow_name ?? "a workflow";
  if (isCredentialItem(item)) {
    const provider = item.providers[0];
    return {
      title: `${provider ? formatProviderName(provider) : "A service"} for ${item.expert_name}`,
      detail: `${workflowName} needs it to run on schedule.`,
    };
  }
  return {
    title: `Schedule ${workflowName} for ${item.expert_name}`,
    detail:
      item.resolution === "inputs"
        ? `Needs ${formatInputList(item.missing_inputs ?? [])} before it can run on schedule.`
        : "Every connection is in place; the schedule still has to be created.",
  };
}

function formatInputList(names: string[]) {
  if (names.length === 0) return "a few details from you";
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}
