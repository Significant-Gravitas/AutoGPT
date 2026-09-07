import { useGetV2ListMarketplaceSkills } from "@/app/api/__generated__/endpoints/store/store";

const SHELF_SIZE = 6;

export function useSkillsSection() {
  const query = useGetV2ListMarketplaceSkills(
    { page_size: SHELF_SIZE },
    {
      query: {
        select: (response) =>
          response.status === 200 ? response.data.skills : [],
      },
    },
  );

  return {
    skills: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
