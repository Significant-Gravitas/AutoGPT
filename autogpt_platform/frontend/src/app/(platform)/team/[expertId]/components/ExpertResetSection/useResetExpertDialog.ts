import { getGetV2ListSessionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetExpertQueryKey,
  useResetExpert,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import { forgetKickoff } from "@/app/(platform)/copilot/expertKickoff";
import { SESSION_LIST_QUERY_KEY } from "@/app/(platform)/copilot/useSessionList";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";

interface Args {
  expertId: string;
  expertName: string;
  onClose: () => void;
}

export function useResetExpertDialog({ expertId, expertName, onClose }: Args) {
  const queryClient = useQueryClient();
  const userId = useAuthStore((state) => state.user?.id) ?? null;

  const { mutate, isPending: isResetting } = useResetExpert({
    mutation: {
      onSuccess: async () => {
        if (userId) forgetKickoff(userId, expertId);
        await Promise.all([
          invalidateExpertRosterQueries(queryClient),
          queryClient.invalidateQueries({
            queryKey: getGetExpertQueryKey(expertId),
          }),
          queryClient.invalidateQueries({
            queryKey: getGetHomeDashboardQueryKey(),
          }),
          queryClient.invalidateQueries({
            queryKey: getGetV2ListSessionsQueryKey(),
          }),
          queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY }),
          invalidateAllScheduleQueries(queryClient),
        ]);
        toast({
          title: `${expertName} is reset`,
          description: `${expertName} will introduce themselves again the next time you chat.`,
        });
        onClose();
      },
      onError: () => {
        toast({
          title: `Could not finish resetting ${expertName}`,
          description: "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  function handleReset() {
    mutate({ expertId });
  }

  return { isResetting, handleReset };
}
