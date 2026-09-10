import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { useGetV2ListMarketplaceSkills } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { SHELF_SIZE } from "./helpers";

export function useSkillsSection() {
  const { isLoggedIn } = useAuth();

  const query = useGetV2ListMarketplaceSkills(
    { page_size: SHELF_SIZE },
    { query: { select: (response) => okData(response) } },
  );

  // An install lands under the listing's slug, so the user's own skill names
  // are what says which shelf cards are already added.
  const installed = useListCopilotSkills({
    query: { select: (res) => okData(res) ?? [], enabled: isLoggedIn },
  });

  return {
    isLoggedIn,
    skills: query.data?.skills ?? [],
    total: query.data?.pagination.total_items ?? 0,
    installedSlugs: new Set((installed.data ?? []).map((skill) => skill.name)),
    isLoading: query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
  };
}
