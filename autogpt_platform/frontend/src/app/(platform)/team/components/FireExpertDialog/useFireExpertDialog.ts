import {
  getListExpertsQueryKey,
  useArchiveExpert,
  useGetExpertDetachPreview,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
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
        queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() });
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
    mutate({ expertId });
  }

  return {
    preview: previewQuery.data ?? null,
    isFiring,
    handleFire,
  };
}
