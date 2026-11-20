import {
  getListExpertsQueryKey,
  useHireExpert,
  useResumeExpertSchedules,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export function useExpertTeamCard(expert: Expert) {
  const expertId = expert.id;
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

  // Re-hiring the template is idempotent and restarts a failed setup.
  const { mutate: rehire, isPending: isRetryingSetup } = useHireExpert({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() });
      },
      onError: () => {
        toast({ title: "Could not retry setup", variant: "destructive" });
      },
    },
  });

  function retrySetup() {
    if (!expert.source_template_id) return;
    rehire({ data: { template_id: expert.source_template_id } });
  }

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
    handleResume,
    isResuming,
    retrySetup,
    isRetryingSetup,
    isFireOpen,
    openFire,
    closeFire,
  };
}
