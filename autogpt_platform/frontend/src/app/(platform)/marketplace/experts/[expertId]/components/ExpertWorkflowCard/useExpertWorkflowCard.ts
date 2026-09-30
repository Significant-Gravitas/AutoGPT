import { useGetV2GetAgentByVersion } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";

/** The marketplace listing behind a preloaded workflow, for its preview
 *  image. Workflows installed from a library agent have no listing. */
export function useExpertWorkflowCard(storeListingVersionId: string | null) {
  const listingQuery = useGetV2GetAgentByVersion(storeListingVersionId ?? "", {
    query: {
      enabled: Boolean(storeListingVersionId),
      select: (response) => okData(response)?.agent_image?.[0] ?? null,
    },
  });

  return {
    imageUrl: listingQuery.data ?? null,
    isLoadingImage: listingQuery.isLoading,
  };
}
