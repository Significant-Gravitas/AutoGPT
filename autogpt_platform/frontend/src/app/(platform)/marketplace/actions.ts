"use server";

import BackendAPI, {
  CreatorsResponse,
  StoreAgentsResponse,
} from "@/lib/autogpt-server-api";

const EMPTY_AGENTS_RESPONSE: StoreAgentsResponse = {
  agents: [],
  pagination: {
    total_items: 0,
    total_pages: 0,
    current_page: 0,
    page_size: 0,
  },
};
const EMPTY_CREATORS_RESPONSE: CreatorsResponse = {
  creators: [],
  pagination: {
    total_items: 0,
    total_pages: 0,
    current_page: 0,
    page_size: 0,
  },
};

export async function getMarketplaceData(): Promise<{
  featuredAgents: StoreAgentsResponse;
  topAgents: StoreAgentsResponse;
  featuredCreators: CreatorsResponse;
}> {
  const api = new BackendAPI();

  const [featuredAgents, topAgents, featuredCreators] = await Promise.all([
    api.getStoreAgents({ featured: true }).catch((error) => {
      console.error("Error fetching featured marketplace agents:", error);
      return EMPTY_AGENTS_RESPONSE;
    }),
    api.getStoreAgents({ sorted_by: "runs" }).catch((error) => {
      console.error("Error fetching top marketplace agents:", error);
      return EMPTY_AGENTS_RESPONSE;
    }),
    api
      .getStoreCreators({ featured: true, sorted_by: "num_agents" })
      .catch((error) => {
        console.error("Error fetching featured marketplace creators:", error);
        return EMPTY_CREATORS_RESPONSE;
      }),
  ]);

  return {
    featuredAgents,
    topAgents,
    featuredCreators,
  };
}
