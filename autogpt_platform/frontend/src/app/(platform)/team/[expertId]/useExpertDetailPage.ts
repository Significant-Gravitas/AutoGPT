import {
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  useGetExpert,
  useResumeExpertSchedules,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getExpertSchedules } from "../helpers";

interface Args {
  expertId: string;
  enabled: boolean;
}

export function useExpertDetailPage({ expertId, enabled }: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isPickerOpen, setIsPickerOpen] = useState(false);

  const expertQuery = useGetExpert(expertId, {
    query: { select: (res) => okData(res) ?? null, enabled },
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], enabled },
  });

  const expert = expertQuery.data ?? null;
  const schedules = expert
    ? getExpertSchedules(expert, schedulesQuery.data ?? [])
    : [];

  const { mutate: resumeSchedules, isPending: isResuming } =
    useResumeExpertSchedules({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() });
          queryClient.invalidateQueries({
            queryKey: getGetExpertQueryKey(expertId),
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

  async function refetch() {
    await Promise.all([expertQuery.refetch(), schedulesQuery.refetch()]);
  }

  return {
    expert,
    isLoading: enabled && (expertQuery.isLoading || schedulesQuery.isLoading),
    isError:
      expertQuery.isError ||
      schedulesQuery.isError ||
      (expertQuery.isFetched && expert === null),
    refetch,
    schedules,
    isPickerOpen,
    openPicker: () => setIsPickerOpen(true),
    closePicker: () => setIsPickerOpen(false),
    resumeSchedules: () => resumeSchedules({ expertId }),
    isResuming,
  };
}
