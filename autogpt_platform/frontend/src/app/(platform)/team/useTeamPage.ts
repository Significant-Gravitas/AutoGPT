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
  const [soulDrawerKey, setSoulDrawerKey] = useState(0);

  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[], enabled },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );

  const schedulesStatus = schedulesQuery.isError
    ? ("error" as const)
    : schedulesQuery.isSuccess
      ? ("loaded" as const)
      : ("loading" as const);

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

  function retrySchedules() {
    void schedulesQuery.refetch();
  }

  function closeSoul() {
    setSoulExpertId(null);
  }

  function openSoul(expertId: string) {
    setSoulExpertId(expertId);
    setSoulDrawerKey((current) => current + 1);
  }

  return {
    hiredExperts,
    schedulesForExpert,
    // Gate loading/error on the primary experts query only; the schedules
    // query keeps its own status so cards can distinguish "no schedules"
    // from "schedules still loading / unavailable" without hiding the roster.
    isLoading: enabled && expertsQuery.isPending,
    isError: expertsQuery.isError,
    schedulesStatus,
    retrySchedules,
    refetch,
    installWorkflow,
    pickerExpertId,
    closeWorkflowPicker,
    soulExpert:
      hiredExperts.find((expert) => expert.id === soulExpertId) ?? null,
    soulDrawerKey,
    openSoul,
    closeSoul,
  };
}
