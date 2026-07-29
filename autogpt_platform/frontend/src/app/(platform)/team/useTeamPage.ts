import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { useState } from "react";

interface Args {
  enabled: boolean;
}

export function useTeamPage({ enabled }: Args) {
  const [pickerExpertId, setPickerExpertId] = useState<string | null>(null);
  const [profileExpertId, setProfileExpertId] = useState<string | null>(null);

  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[], enabled },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );

  function installWorkflow(expertId: string) {
    setPickerExpertId(expertId);
  }

  function closeWorkflowPicker() {
    setPickerExpertId(null);
  }

  return {
    hiredExperts,
    isLoading: enabled && expertsQuery.isLoading,
    isError: expertsQuery.isError,
    refetch: expertsQuery.refetch,
    installWorkflow,
    pickerExpertId,
    closeWorkflowPicker,
    profileExpert:
      hiredExperts.find((expert) => expert.id === profileExpertId) ?? null,
    openProfile: setProfileExpertId,
    closeProfile: () => setProfileExpertId(null),
  };
}
