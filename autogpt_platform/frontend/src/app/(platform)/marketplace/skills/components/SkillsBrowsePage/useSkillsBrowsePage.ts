import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { useGetV2ListMarketplaceSkillsInfinite } from "@/app/api/__generated__/endpoints/store/store";
import {
  getPaginationNextPageNumber,
  okData,
  unpaginate,
} from "@/app/api/helpers";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useRouter, useSearchParams } from "next/navigation";
import { BROWSE_PAGE_SIZE } from "../../../components/SkillsSection/helpers";

export function useSkillsBrowsePage() {
  const router = useRouter();
  const params = useSearchParams();
  const { isLoggedIn } = useAuth();

  const search = params.get("q") ?? "";
  const category = params.get("category");

  const query = useGetV2ListMarketplaceSkillsInfinite(
    {
      page: 1,
      page_size: BROWSE_PAGE_SIZE,
      ...(search ? { search_query: search } : {}),
      ...(category ? { category } : {}),
    },
    { query: { getNextPageParam: getPaginationNextPageNumber } },
  );

  const installed = useListCopilotSkills({
    query: { select: (res) => okData(res) ?? [], enabled: isLoggedIn },
  });

  // Both filters live in the URL so a filtered shelf can be linked and the
  // back button steps through them.
  function push(next: { search: string; category: string | null }) {
    const qs = new URLSearchParams();
    if (next.search) qs.set("q", next.search);
    if (next.category) qs.set("category", next.category);
    const query = qs.toString();
    router.push(query ? `/marketplace/skills?${query}` : "/marketplace/skills");
  }

  return {
    search,
    category,
    skills: query.data ? unpaginate(query.data, "skills") : [],
    total: okData(query.data?.pages[0])?.pagination.total_items ?? 0,
    installedSlugs: new Set((installed.data ?? []).map((skill) => skill.name)),
    isLoading: query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
    hasMore: Boolean(query.hasNextPage),
    isFetchingMore: query.isFetchingNextPage,
    loadMore: query.fetchNextPage,
    setSearch: (value: string) => push({ search: value, category }),
    setCategory: (value: string | null) => push({ search, category: value }),
  };
}
