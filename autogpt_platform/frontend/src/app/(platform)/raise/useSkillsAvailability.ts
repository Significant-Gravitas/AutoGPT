import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { useGetV2ListMarketplaceSkills } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { MAX_SEARCH_RESULTS } from "./components/KitStep/helpers";

// The skills beat is only worth asking when there is something to add: a live
// Hub listing, or a skill the user has already written. Both queries share
// their keys with the picker's own search, so asking costs no extra request.
export function useSkillsAvailability() {
  const hub = useFlagStatus(Flag.SKILLS_HUB);
  const isHubOn = hub.ready && hub.enabled;

  const marketplace = useGetV2ListMarketplaceSkills(
    { search_query: "", page_size: MAX_SEARCH_RESULTS },
    {
      query: {
        enabled: isHubOn,
        select: (response) => okData(response)?.skills.length ?? 0,
      },
    },
  );
  const library = useListCopilotSkills(undefined, {
    query: { select: (response) => okData(response)?.length ?? 0 },
  });

  const isReady =
    hub.ready && !library.isLoading && (!isHubOn || !marketplace.isLoading);

  return {
    isReady,
    // An unsettled answer keeps the beat. It is the flow's last step, so
    // dropping it moves the submit button — better late than flickering.
    hasSkillsToOffer:
      !isReady || (marketplace.data ?? 0) > 0 || (library.data ?? 0) > 0,
  };
}
