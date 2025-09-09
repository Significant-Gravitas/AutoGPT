import { unstable_cache } from "next/cache";
import {
  getV2ListStoreAgents,
  getV2ListStoreCreators,
} from "@/app/api/__generated__/endpoints/store/store";
import { SearchResultsClient } from "./SearchResultsClient";
import { StoreAgent, Creator } from "@/lib/autogpt-server-api";

// Cache the search results for 10 minutes based on search term and sort
const getCachedSearchResults = unstable_cache(
  async (searchTerm: string, sort: string) => {
    try {
      const [agentsRes, creatorsRes] = await Promise.all([
        getV2ListStoreAgents({
          search_query: searchTerm,
          sorted_by: sort,
        }),
        getV2ListStoreCreators({
          search_query: searchTerm,
        }),
      ]);

      return {
        agents: ('agents' in agentsRes.data ? agentsRes.data.agents || [] : []) as StoreAgent[],
        creators: ('creators' in creatorsRes.data ? creatorsRes.data.creators || [] : []) as Creator[],
      };
    } catch (error) {
      console.error("Error fetching search results:", error);
      return {
        agents: [] as StoreAgent[],
        creators: [] as Creator[],
      };
    }
  },
  ["marketplace-search"], // Cache key prefix
  {
    revalidate: 600, // 10 minutes
    tags: ["marketplace-search"],
  },
);

export async function SearchResults({
  searchTerm,
  sort,
}: {
  searchTerm: string;
  sort: string;
}) {
  const { agents, creators } = await getCachedSearchResults(searchTerm, sort);

  const agentsCount = agents.length;
  const creatorsCount = creators.length;
  const totalCount = agentsCount + creatorsCount;

  if (totalCount === 0) {
    return (
      <div className="mt-20 flex flex-col items-center justify-center">
        <h3 className="mb-2 text-xl font-medium text-neutral-600 dark:text-neutral-300">
          No results found
        </h3>
        <p className="text-neutral-500 dark:text-neutral-400">
          Try adjusting your search terms or filters
        </p>
      </div>
    );
  }

  return (
    <SearchResultsClient
      initialAgents={agents}
      initialCreators={creators}
      agentsCount={agentsCount}
      creatorsCount={creatorsCount}
      totalCount={totalCount}
    />
  );
}
