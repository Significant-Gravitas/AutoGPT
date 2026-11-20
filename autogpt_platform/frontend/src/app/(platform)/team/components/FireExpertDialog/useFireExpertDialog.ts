import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import {
  getGetExpertQueryKey,
  useArchiveExpert,
  useGetExpertDetachPreview,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";

interface Args {
  expertId: string;
  expertName: string;
  open: boolean;
  onClose: () => void;
  onFired?: () => void;
}

export function useFireExpertDialog({
  expertId,
  expertName,
  open,
  onClose,
  onFired,
}: Args) {
  const queryClient = useQueryClient();

  const previewQuery = useGetExpertDetachPreview(expertId, {
    query: { enabled: open, select: (res) => okData(res) ?? null },
  });

  const { mutate, isPending: isFiring } = useArchiveExpert({
    mutation: {
      onSuccess: async () => {
        await Promise.all([
          invalidateExpertRosterQueries(queryClient),
          queryClient.invalidateQueries({
            queryKey: getGetExpertQueryKey(expertId),
          }),
          queryClient.invalidateQueries({
            queryKey: getGetHomeDashboardQueryKey(),
          }),
          invalidateAllScheduleQueries(queryClient),
        ]);
        toast({
          title: `${expertName} was fired`,
          description: `You can re-hire ${expertName} anytime from the marketplace.`,
        });
        onClose();
        onFired?.();
      },
      onError: () => {
        toast({
          title: `Could not fire ${expertName}`,
          description: `${expertName} is still on your team. Please try again.`,
          variant: "destructive",
        });
      },
    },
  });

  function handleFire() {
    // Any in-flight preview blocks firing, including a manual retry with
    // cached data. A failed, settled preview remains informational and does
    // not hard-block the destructive action.
    if (previewQuery.isFetching) return;
    mutate({ expertId });
  }

  return {
    preview: previewQuery.data ?? null,
    isPreviewLoading: previewQuery.isFetching,
    isPreviewError: previewQuery.isError,
    isPreviewReady: previewQuery.isSuccess,
    retryPreview: previewQuery.refetch,
    isFiring,
    handleFire,
  };
}
