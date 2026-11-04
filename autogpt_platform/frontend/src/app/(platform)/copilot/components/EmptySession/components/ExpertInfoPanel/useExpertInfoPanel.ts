import {
  useGetExpert,
  useListExpertCredentials,
} from "@/app/api/__generated__/endpoints/experts/experts";
import {
  useGetV1ListExecutionSchedulesForAUser,
  useListCopilotFollowupSchedules,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";
import { useTaskList } from "@/app/(platform)/tasks/data/useTasksData";
import { getExpertSchedules } from "@/app/(platform)/team/helpers";
import { useState } from "react";
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
  const tasksQuery = useTaskList(
    { assignee_ids: expertId ? [expertId] : [] },
    { enabled: isOpen },
  );

  const expert = expertQuery.data ?? null;
  const followups = (followupsQuery.data ?? []).filter(
    (followup) => (followup.expert_id ?? null) === expertId,
  );
  const graphSchedules = expert
    ? getExpertSchedules(expert, graphSchedulesQuery.data ?? [])
    : [];

  return {
    expert,
    isLoading: expertQuery.isLoading,
    schedules: toPanelSchedules(graphSchedules, followups),
    isLoadingSchedules:
      followupsQuery.isLoading || (enabled && graphSchedulesQuery.isLoading),
    credentials: credentialsQuery.data ?? [],
    isLoadingCredentials: credentialsQuery.isLoading,
    tasks: (tasksQuery.data ?? []).slice(
      0,
      expertId ? EXPERT_TASKS : TEAM_TASKS,
    ),
    isLoadingTasks: tasksQuery.isLoading,
    tab,
    setTab,
  };
}
