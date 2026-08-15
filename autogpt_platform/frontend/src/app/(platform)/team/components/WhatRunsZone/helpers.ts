import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import {
  getExpertSchedules,
  getGraphWorkflowCounts,
  getWorkflowSchedules,
} from "../../helpers";

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
    const graphWorkflowCounts = getGraphWorkflowCounts(expert.workflows);
    const workflows = expert.workflows.map((workflow) => ({
      workflow,
      schedules: getWorkflowSchedules(
        workflow,
        expertSchedules,
        graphWorkflowCounts,
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

/** Keep agents that still have at least one eligible expert target. Matching
 *  the immutable listing-version id lets a newer snapshot remain adoptable
 *  and lets one agent be installed on more than one expert. */
export function getUnadoptedAgents(
  agents: LibraryAgent[],
  experts: Expert[],
  adoptedTargetKeys?: ReadonlySet<string>,
): LibraryAgent[] {
  return agents.filter(
    (agent) =>
      !getAdoptTargetVersionID(agent) ||
      getAdoptableExperts(agent, experts, adoptedTargetKeys).length > 0,
  );
}

export function getAdoptableExperts(
  agent: LibraryAgent,
  experts: Expert[],
  adoptedTargetKeys?: ReadonlySet<string>,
): Expert[] {
  const versionID = getAdoptTargetVersionID(agent);
  if (!versionID) return experts;
  return experts.filter(
    (expert) =>
      !expert.workflows.some(
        (workflow) => workflow.store_listing_version_id === versionID,
      ) && !adoptedTargetKeys?.has(getAdoptTargetKey(agent, expert)),
  );
}

export function getAdoptTargetKey(agent: LibraryAgent, expert: Expert) {
  return `${agent.id}:${expert.id}`;
}

function parseAdoptTargetKey(key: string) {
  const separator = key.lastIndexOf(":");
  if (separator < 1 || separator === key.length - 1) return null;
  return {
    agentID: key.slice(0, separator),
    expertID: key.slice(separator + 1),
  };
}

export function pruneAdoptedTargetKeys(
  adoptedTargetKeys: Set<string>,
  agents: LibraryAgent[],
  experts: Expert[],
) {
  if (adoptedTargetKeys.size === 0) return adoptedTargetKeys;

  const agentsByID = new Map(agents.map((agent) => [agent.id, agent]));
  const expertsByID = new Map(experts.map((expert) => [expert.id, expert]));
  const next = new Set<string>();
  for (const key of adoptedTargetKeys) {
    const target = parseAdoptTargetKey(key);
    if (!target) continue;
    const agent = agentsByID.get(target.agentID);
    const expert = expertsByID.get(target.expertID);
    if (!agent || !expert) continue;
    const versionID = getAdoptTargetVersionID(agent);
    if (!versionID) continue;
    const isConfirmed = expert.workflows.some(
      (workflow) => workflow.store_listing_version_id === versionID,
    );
    if (!isConfirmed) next.add(key);
  }

  return next.size === adoptedTargetKeys.size ? adoptedTargetKeys : next;
}

/** The immutable marketplace version matching this agent's exact graph
 *  snapshot, resolved server-side. Pure-local agents have none, so Adopt is
 *  hidden for them — the install endpoint only accepts a
 *  store_listing_version_id. */
export function getAdoptTargetVersionID(agent: LibraryAgent) {
  return agent.store_listing_version_id;
}
