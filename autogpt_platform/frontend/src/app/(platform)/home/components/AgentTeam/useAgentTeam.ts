import { useState } from "react";
import {
  useListExpertPods,
  useListExperts,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { okData } from "@/app/api/helpers";
import { buildPodRollups, POD_ROLLUP_THRESHOLD } from "./helpers";

export function useAgentTeam(agents: HomeAgentStatus[]) {
  const [expandedPodIds, setExpandedPodIds] = useState<string[]>([]);
  const shouldCollapse = agents.length > POD_ROLLUP_THRESHOLD;

  const expertsQuery = useListExperts({
    query: {
      enabled: shouldCollapse,
      select: (response) => (okData(response) ?? []) as Expert[],
    },
  });
  const podsQuery = useListExpertPods({
    query: {
      enabled: shouldCollapse,
      select: (response) => (okData(response) ?? []) as ExpertPod[],
    },
  });
  const tasksQuery = useListTasks(undefined, {
    query: {
      enabled: shouldCollapse,
      select: (response) => (okData(response) ?? []) as DelegatedTask[],
    },
  });

  const showRollups =
    shouldCollapse &&
    expertsQuery.data !== undefined &&
    podsQuery.data !== undefined &&
    tasksQuery.data !== undefined;

  const rollups = showRollups
    ? buildPodRollups({
        agents,
        experts: expertsQuery.data,
        pods: podsQuery.data,
        tasks: tasksQuery.data,
      })
    : [];

  function togglePod(podId: string) {
    setExpandedPodIds((current) =>
      current.includes(podId)
        ? current.filter((id) => id !== podId)
        : [...current, podId],
    );
  }

  return { showRollups, rollups, expandedPodIds, togglePod };
}
