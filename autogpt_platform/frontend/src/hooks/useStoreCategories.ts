import { useGetV2ListStoreCategories } from "@/app/api/__generated__/endpoints/store/store";
import { StoreCategoryInfo } from "@/app/api/__generated__/models/storeCategoryInfo";

export function useStoreCategories() {
  const { data, isLoading } = useGetV2ListStoreCategories({
    query: {
      // The canonical set only changes with a deploy.
      staleTime: Infinity,
      select: (x) => x.data as StoreCategoryInfo[],
    },
  });

  return { categories: data ?? [], isLoading };
}
