import type { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";

export const POD_ROLLUP_THRESHOLD = 5;
export const UNASSIGNED_POD_ID = "unassigned";

export interface PodRollup {
  id: string;
  name: string;
  agents: HomeAgentStatus[];
  activeCount: number;
  needsYouCount: number;
  spendCents: number;
}

interface Args {
  agents: HomeAgentStatus[];
  experts: Expert[];
  pods: ExpertPod[];
  tasks: DelegatedTask[];
}

export function buildPodRollups({ agents, experts, pods, tasks }: Args) {
  const podIdByExpert = new Map(
    experts.map((expert) => [expert.id, expert.pod_id ?? UNASSIGNED_POD_ID]),
  );
  const rollups = new Map<string, PodRollup>(
    [...pods, { id: UNASSIGNED_POD_ID, name: "Unassigned" }].map((pod) => [
      pod.id,
      {
        id: pod.id,
        name: pod.name,
        agents: [],
        activeCount: 0,
        needsYouCount: 0,
        spendCents: 0,
      },
    ]),
  );

  for (const agent of agents) {
    const rollup = rollupFor(rollups, podIdByExpert.get(agent.expert.id));
    rollup.agents.push(agent);
    rollup.spendCents += agent.spend_cents ?? 0;
  }

  for (const task of tasks) {
    if (!task.owner) continue;
    const podId = podIdByExpert.get(task.owner.id);
    if (podId === undefined) continue;
    const rollup = rollupFor(rollups, podId);
    if (task.status === "WAITING_USER") rollup.needsYouCount += 1;
    else if (task.status === "WORKING" || task.status === "QUEUED") {
      rollup.activeCount += 1;
    }
  }

  return [...rollups.values()].filter((rollup) => rollup.agents.length > 0);
}

function rollupFor(
  rollups: Map<string, PodRollup>,
  podId: string | undefined,
): PodRollup {
  return (
    rollups.get(podId ?? UNASSIGNED_POD_ID) ??
    (rollups.get(UNASSIGNED_POD_ID) as PodRollup)
  );
}
