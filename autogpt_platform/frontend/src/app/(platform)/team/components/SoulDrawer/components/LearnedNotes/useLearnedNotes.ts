import {
  getListMyExpertMemoryFactsQueryKey,
  useForgetMyExpertMemoryFact,
  useListMyExpertMemoryFacts,
} from "@/app/api/__generated__/endpoints/memory/memory";
import { toast } from "@/components/molecules/Toast/use-toast";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";

/** Enough to show what the expert is picking up without crowding out the Soul
 *  fields above it; the rest is one click away on the memory page. */
export const SOUL_NOTES_LIMIT = 5;

export function useLearnedNotes(expertId: string) {
  const isMemoryEnabled = useGetFlag(Flag.GRAPHITI_MEMORY);
  const queryClient = useQueryClient();

  const factsQuery = useListMyExpertMemoryFacts(
    expertId,
    { limit: SOUL_NOTES_LIMIT },
    { query: { enabled: Boolean(isMemoryEnabled) } },
  );
  const facts =
    factsQuery.data?.status === 200 ? factsQuery.data.data.items : [];

  const forgetFact = useForgetMyExpertMemoryFact({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getListMyExpertMemoryFactsQueryKey(expertId),
        });
        toast({ title: "Forgotten" });
      },
      onError: () => {
        toast({
          title: "Could not forget that memory",
          description: "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  async function forget(uuid: string) {
    try {
      await forgetFact.mutateAsync({ expertId, factUuid: uuid });
    } catch {
      // onError already surfaced the toast; keep the rejection out of the
      // click handler.
    }
  }

  return {
    facts,
    isLoading: factsQuery.isLoading,
    isError: factsQuery.isError,
    forget,
    forgettingUuid: forgetFact.isPending
      ? (forgetFact.variables?.factUuid ?? null)
      : null,
  };
}
