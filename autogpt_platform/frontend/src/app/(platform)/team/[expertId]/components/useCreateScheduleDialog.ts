import { getGetExpertQueryKey } from "@/app/api/__generated__/endpoints/experts/experts";
import { getGetV1ListExecutionSchedulesForAUserQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
export function useCreateScheduleDialog(expertId: string, onClose: () => void) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  async function handleScheduleCreated() {
    await Promise.all([
      queryClient.invalidateQueries({
        queryKey: getGetV1ListExecutionSchedulesForAUserQueryKey(),
      }),
      queryClient.invalidateQueries({
        queryKey: getGetExpertQueryKey(expertId),
      }),
    ]);
    toast({ title: "Schedule created", variant: "success" });
    onClose();
  }

  return { handleScheduleCreated };
}
