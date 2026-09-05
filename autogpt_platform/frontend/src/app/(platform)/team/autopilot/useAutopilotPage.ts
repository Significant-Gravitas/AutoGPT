import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import { getHiredExperts, getTeamSchedules, getTeamSkills } from "../helpers";

interface Args {
  enabled: boolean;
}

export function useAutopilotPage({ enabled }: Args) {
  const expertsQuery = useListExperts({
    query: { select: (res) => (okData(res) ?? []) as Expert[], enabled },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled },
  });

  const experts = getHiredExperts(expertsQuery.data ?? []);
  const schedules = getTeamSchedules(experts, schedulesQuery.data ?? []);
  const skills = getTeamSkills(experts);

  function refetch() {
    return Promise.all([expertsQuery.refetch(), schedulesQuery.refetch()]);
  }

  return {
    experts,
    schedules,
    skills,
    isLoading: enabled && (expertsQuery.isLoading || schedulesQuery.isLoading),
    isError: expertsQuery.isError || schedulesQuery.isError,
    refetch,
  };
}
