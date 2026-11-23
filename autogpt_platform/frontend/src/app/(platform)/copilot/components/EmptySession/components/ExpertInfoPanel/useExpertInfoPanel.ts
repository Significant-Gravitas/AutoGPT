import {
  useGetExpert,
  useListExpertCredentials,
} from "@/app/api/__generated__/endpoints/experts/experts";
import {
  useGetV1ListExecutionSchedulesForAUser,
  useListCopilotFollowupSchedules,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { okData } from "@/app/api/helpers";
import { getExpertSchedules } from "@/app/(platform)/team/helpers";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { useState } from "react";
import { useExpertMap } from "../../../../useExpertMap";
import { toPanelSchedules } from "./helpers";

export type InfoTab = "details" | "access" | "tasks";

const EXPERT_TASKS = 8;
const TEAM_TASKS = 5;

export function useExpertInfoPanel(expertId: string | null, isOpen: boolean) {
  const [tab, setTab] = useState<InfoTab>("details");
  const enabled = isOpen && !!expertId;

  const expertQuery = useGetExpert(expertId ?? "", {
    query: { select: (res) => okData(res) ?? null, enabled },
  });
  const graphSchedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled },
  });
  const followupsQuery = useListCopilotFollowupSchedules({
    query: { select: (res) => okData(res) ?? [], enabled: isOpen },
  });
  const credentialsQuery = useListExpertCredentials(expertId ?? "", {
    query: { select: (res) => okData(res) ?? [], enabled },
  });
  // Otto's panel lists the whole team's latest tasks; an expert's, their own.
  const tasksQuery = useListTasks(
    expertId
      ? { owner_expert_id: expertId, limit: EXPERT_TASKS }
      : { limit: TEAM_TASKS },
    { query: { select: (res) => okData(res)?.tasks ?? [], enabled: isOpen } },
  );
  const { expertsById } = useExpertMap();

  const expert = expertQuery.data ?? null;
  const followups = (followupsQuery.data ?? []).filter(
    (followup) => (followup.expert_id ?? null) === expertId,
  );
  const graphSchedules = expert
    ? getExpertSchedules(expert, graphSchedulesQuery.data ?? [])
    : [];

  function ownerName(task: Task) {
    if (!task.owner_expert_id) return AUTOPILOT_NAME;
    return expertsById.get(task.owner_expert_id)?.name ?? "An expert";
  }

  return {
    expert,
    isLoading: expertQuery.isLoading,
    schedules: toPanelSchedules(graphSchedules, followups),
    isLoadingSchedules:
      followupsQuery.isLoading || (enabled && graphSchedulesQuery.isLoading),
    credentials: credentialsQuery.data ?? [],
    isLoadingCredentials: credentialsQuery.isLoading,
    tasks: tasksQuery.data ?? [],
    isLoadingTasks: tasksQuery.isLoading,
    ownerName,
    tab,
    setTab,
  };
}
