import { useGetV2GetMarketplaceSkill } from "@/app/api/__generated__/endpoints/store/store";

export function useSkillPage(slug: string) {
  const query = useGetV2GetMarketplaceSkill(slug, {
    query: {
      select: (response) => (response.status === 200 ? response.data : null),
    },
  });

  return {
    skill: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
