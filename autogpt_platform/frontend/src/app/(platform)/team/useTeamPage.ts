import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import { useState } from "react";
import { getExpertSchedules } from "./helpers";

interface Args {
  enabled: boolean;
}

export function useTeamPage({ enabled }: Args) {
  const [pickerExpertId, setPickerExpertId] = useState<string | null>(null);
  const [soulExpertId, setSoulExpertId] = useState<string | null>(null);

  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[], enabled },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );

  function schedulesForExpert(expert: Expert) {
    return getExpertSchedules(expert, schedulesQuery.data ?? []);
  }

  function installWorkflow(expertId: string) {
    setPickerExpertId(expertId);
  }

  function closeWorkflowPicker() {
    setPickerExpertId(null);
  }

  function refetch() {
    return Promise.all([expertsQuery.refetch(), schedulesQuery.refetch()]);
  }

  function closeSoul() {
    setSoulExpertId(null);
  }


  return {
    hiredExperts,
    schedulesForExpert,
    isLoading: enabled && (expertsQuery.isLoading || schedulesQuery.isLoading),
    isError: expertsQuery.isError || schedulesQuery.isError,
    refetch,
    installWorkflow,
    pickerExpertId,
    closeWorkflowPicker,
    soulExpert:
      hiredExperts.find((expert) => expert.id === soulExpertId) ?? null,
    openSoul: setSoulExpertId,
    closeSoul,
  };
}
