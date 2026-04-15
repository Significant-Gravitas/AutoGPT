import {
  usePostV2ResetCopilotUsage,
  getGetV2GetCopilotUsageQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api";
import { useQueryClient } from "@tanstack/react-query";
import { useRef } from "react";

export function useResetRateLimit(options?: {
  onSuccess?: () => void;
  onCreditChange?: () => void;
}) {
  // Use refs so mutation callbacks always see the latest options,
  // avoiding stale-closure issues when the caller re-renders with
  // different callback references.
  const onSuccessRef = useRef(options?.onSuccess);
  onSuccessRef.current = options?.onSuccess;
  const onCreditChangeRef = useRef(options?.onCreditChange);
  onCreditChangeRef.current = options?.onCreditChange;

  const queryClient = useQueryClient();
  const { mutate: resetUsage, isPending } = usePostV2ResetCopilotUsage({
    mutation: {
      onSuccess: async () => {
        // Await the usage refetch so the UI shows updated limits before
        // closing the dialog or re-enabling the reset CTA.
        // invalidateQueries already triggers a refetch for active queries.
        await queryClient.invalidateQueries({
          queryKey: getGetV2GetCopilotUsageQueryKey(),
        });
        onCreditChangeRef.current?.();
        toast({
          title: "Rate limit reset",
          description:
            "Your daily usage limit has been reset. You can continue working.",
        });
        onSuccessRef.current?.();
      },
      onError: (error: unknown) => {
        const message =
          error instanceof ApiError
            ? (error.response?.detail ?? error.message)
            : error instanceof Error
              ? error.message
              : "Failed to reset limit.";
        toast({
          title: "Reset failed",
          description: message,
          variant: "destructive",
        });
      },
    },
  });

  return { resetUsage, isPending };
}
