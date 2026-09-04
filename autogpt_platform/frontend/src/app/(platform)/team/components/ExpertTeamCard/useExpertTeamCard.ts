import {
  getListExpertsQueryKey,
  useListExpertCredentials,
  useResumeExpertSchedules,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export function useExpertTeamCard(expertId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isFireOpen, setIsFireOpen] = useState(false);
  const credentialsQuery = useListExpertCredentials(expertId, {
    query: { select: (response) => okData(response) ?? [] },
  });
  const { mutate: resumeSchedules, isPending: isResuming } =
    useResumeExpertSchedules({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getListExpertsQueryKey(),
          });
        },
        onError: () => {
          toast({
            title: "Could not resume schedules",
            variant: "destructive",
          });
        },
      },
    });

  function handleResume() {
    resumeSchedules({ expertId });
  }

  function openFire() {
    setIsFireOpen(true);
  }

  function closeFire() {
    setIsFireOpen(false);
  }

  return {
    credentialCount: credentialsQuery.data?.length ?? 0,
    handleResume,
    isResuming,
    isFireOpen,
    openFire,
    closeFire,
  };
}
