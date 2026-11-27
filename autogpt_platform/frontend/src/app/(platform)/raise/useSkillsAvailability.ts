import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { useGetV2ListMarketplaceSkills } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { MAX_SEARCH_RESULTS } from "./components/KitStep/helpers";

// The skills beat is only worth asking when there is something to add: a live
// Hub listing, or a skill the user has already written. Both queries share
// their keys with the picker's own search, so asking costs no extra request.
export function useSkillsAvailability() {
  const marketplace = useGetV2ListMarketplaceSkills(
    { search_query: "", page_size: MAX_SEARCH_RESULTS },
    {
      query: {
        select: (response) => okData(response)?.skills.length ?? 0,
      },
    },
  );
  const library = useListCopilotSkills(undefined, {
    query: { select: (response) => okData(response)?.length ?? 0 },
  });

  const isSettled =
    !library.isLoading &&
    !library.isError &&
    !marketplace.isLoading &&
    !marketplace.isError;

  return {
    isReady: isSettled,
    // An unanswered or failed query keeps the beat: a request that did not
    // come back is not an empty catalogue. It is the flow's last step, so
    // dropping it moves the submit button — better late than flickering.
    hasSkillsToOffer:
      !isSettled || (marketplace.data ?? 0) > 0 || (library.data ?? 0) > 0,
  };
}
