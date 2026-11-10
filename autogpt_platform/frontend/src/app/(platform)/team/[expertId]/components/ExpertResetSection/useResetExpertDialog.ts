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
import { useState } from "react";
import { isBusyError, resetFailedToast, type ResetPhase } from "./helpers";

interface Args {
  expertId: string;
  expertName: string;
  open: boolean;
}

export function useResetExpertDialog({ expertId, expertName, open }: Args) {
  const queryClient = useQueryClient();
  const userId = useAuthStore((state) => state.user?.id) ?? null;
  const [phase, setPhase] = useState<ResetPhase>("confirm");
  const [wasOpen, setWasOpen] = useState(open);

  // Each opening starts at the confirm step, whatever the last one ended on.
  if (open !== wasOpen) {
    setWasOpen(open);
    if (open) setPhase("confirm");
  }

  const { mutate, isPending: isResetting } = useResetExpert({
    mutation: {
      onSuccess: async () => {
        if (userId) forgetKickoff(userId, expertId);
        setPhase("done");
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
      },
      onError: (error) => {
        if (isBusyError(error)) {
          setPhase("busy");
          return;
        }
        toast(resetFailedToast(expertName));
      },
    },
  });

  function handleReset() {
    mutate({ expertId });
  }

  return { phase, isResetting, handleReset };
}
