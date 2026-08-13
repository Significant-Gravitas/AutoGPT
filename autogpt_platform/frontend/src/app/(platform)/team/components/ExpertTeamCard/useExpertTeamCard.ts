import {
  getListExpertsQueryKey,
  useResumeExpertSchedules,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export function useExpertTeamCard(expertId: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isFireOpen, setIsFireOpen] = useState(false);
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

  return {
    handleResume,
    isResuming,
    isFireOpen,
    openFire: () => setIsFireOpen(true),
    closeFire: () => setIsFireOpen(false),
  };
}
