import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getExpertSchedules, getWorkflowSchedules } from "../../helpers";

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
  workflows: {
    workflow: ExpertWorkflowRef;
    schedules: GraphExecutionJobInfo[];
  }[];
}

export function getVisibleGroups(
  experts: Expert[],
  schedules: GraphExecutionJobInfo[],
  filter: WhatRunsFilter,
): ExpertWorkflowGroupData[] {
  const view = getFilterView(filter);
  if (!view.showGroups) return [];
  const groups: ExpertWorkflowGroupData[] = [];
  for (const expert of experts) {
    const expertSchedules = getExpertSchedules(expert, schedules);
    const workflows = expert.workflows.map((workflow) => ({
      workflow,
      schedules: getWorkflowSchedules(
        workflow,
        expertSchedules,
        expert.workflows,
      ),
    }));
    const visibleWorkflows = view.scheduledOnly
      ? workflows.filter((item) => item.schedules.length > 0)
      : workflows;
    if (view.includeEmptyGroups || visibleWorkflows.length > 0) {
      groups.push({ expert, workflows: visibleWorkflows });
    }
  }
  return groups;
}

/** Library agents the user owns that are not yet installed on any expert,
 *  matched on the shared graph id that the install flow copies onto the
 *  expert workflow ref. */
export function getUnadoptedAgents(
  agents: LibraryAgent[],
  experts: Expert[],
): LibraryAgent[] {
  const installedGraphIds = new Set<string>();
  for (const expert of experts) {
    for (const workflow of expert.workflows) {
      if (workflow.graph_id) installedGraphIds.add(workflow.graph_id);
    }
  }
  return agents.filter((agent) => !installedGraphIds.has(agent.graph_id));
}

/** The immutable marketplace version matching this agent's exact graph
 *  snapshot, resolved server-side. Pure-local agents have none, so Adopt is
 *  hidden for them — the install endpoint only accepts a
 *  store_listing_version_id. */
export function getAdoptTargetVersionId(agent: LibraryAgent) {
  return agent.store_listing_version_id;
}
