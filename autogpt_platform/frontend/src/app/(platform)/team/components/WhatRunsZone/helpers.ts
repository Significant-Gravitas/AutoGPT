import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";

export type WhatRunsFilter =
  | "all"
  | "members"
  | "agents"
  | "workflows"
  | "scheduled";

export const WHAT_RUNS_FILTERS: { id: WhatRunsFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "members", label: "Members" },
  { id: "agents", label: "Agents" },
  { id: "workflows", label: "Workflows" },
  { id: "scheduled", label: "Scheduled" },
];

interface FilterView {
  showGroups: boolean;
  showAgents: boolean;
  includeEmptyGroups: boolean;
  scheduledOnly: boolean;
}

export function getFilterView(filter: WhatRunsFilter): FilterView {
  switch (filter) {
    case "members":
      return {
        showGroups: true,
        showAgents: false,
        includeEmptyGroups: true,
        scheduledOnly: false,
      };
    case "agents":
      return {
        showGroups: false,
        showAgents: true,
        includeEmptyGroups: false,
        scheduledOnly: false,
      };
    case "workflows":
      return {
        showGroups: true,
        showAgents: false,
        includeEmptyGroups: false,
        scheduledOnly: false,
      };
    case "scheduled":
      return {
        showGroups: true,
        showAgents: false,
        includeEmptyGroups: false,
        scheduledOnly: true,
      };
    case "all":
    default:
      return {
        showGroups: true,
        showAgents: true,
        includeEmptyGroups: true,
        scheduledOnly: false,
      };
  }
}

export interface ExpertWorkflowGroupData {
  expert: Expert;
  workflows: ExpertWorkflowRef[];
}

export function isWorkflowScheduled(workflow: ExpertWorkflowRef) {
  return Boolean(workflow.schedule_id);
}

export function workflowNeedsSetup(workflow: ExpertWorkflowRef) {
  return Boolean(workflow.schedule_cron) && !workflow.schedule_id;
}

export function getWorkflowScheduleLabel(workflow: ExpertWorkflowRef) {
  if (!workflow.schedule_cron) return null;
  return safeHumanizeCron(workflow.schedule_cron);
}

export function getVisibleGroups(
  experts: Expert[],
  filter: WhatRunsFilter,
): ExpertWorkflowGroupData[] {
  const view = getFilterView(filter);
  if (!view.showGroups) return [];
  return experts
    .map((expert) => ({
      expert,
      workflows: view.scheduledOnly
        ? expert.workflows.filter(isWorkflowScheduled)
        : expert.workflows,
    }))
    .filter((group) => view.includeEmptyGroups || group.workflows.length > 0);
}

/** Library agents the user owns that are not yet installed on any expert,
 *  matched on the shared graph id that the install flow copies onto the
 *  expert workflow ref. */
export function getUnadoptedAgents(
  agents: LibraryAgent[],
  experts: Expert[],
): LibraryAgent[] {
  const installedGraphIds = new Set(
    experts
      .flatMap((expert) =>
        expert.workflows.map((workflow) => workflow.graph_id),
      )
      .filter((graphId): graphId is string => Boolean(graphId)),
  );
  return agents.filter((agent) => !installedGraphIds.has(agent.graph_id));
}

/** The immutable marketplace version matching this agent's exact graph
 *  snapshot, resolved server-side. Pure-local agents have none, so Adopt is
 *  hidden for them — the install endpoint only accepts a
 *  store_listing_version_id. */
export function getAdoptTargetVersionId(agent: LibraryAgent): string | null {
  return agent.store_listing_version_id ?? null;
}

export function safeHumanizeCron(cron: string): string {
  try {
    return humanizeCronExpression(cron);
  } catch {
    return cron;
  }
}
