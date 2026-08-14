import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import {
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  useArchiveExpert,
  useGetExpertDetachPreview,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
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
      onSuccess: () => {
        // The paramless list key is a prefix of the archived-inclusive one used
        // by useExpertMap, so a single invalidate refreshes roster, marketplace,
        // sidebar and the chat identity map. The rest of the expert's cached
        // footprint (detail page, home team strip, schedules drawer) would
        // otherwise stay stale for the global 60s window.
        queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() });
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(expertId),
        });
        queryClient.invalidateQueries({
          queryKey: getGetHomeDashboardQueryKey(),
        });
        invalidateAllScheduleQueries(queryClient);
        toast({
          title: `${expertName} was let go`,
          description: `You can re-hire ${expertName} anytime from the marketplace.`,
        });
        onClose();
        onFired?.();
      },
      onError: () => {
        toast({
          title: `Could not fire ${expertName}`,
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
