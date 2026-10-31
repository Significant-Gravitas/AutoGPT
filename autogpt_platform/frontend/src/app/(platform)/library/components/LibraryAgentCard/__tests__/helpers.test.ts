import { describe, expect, test } from "vitest";
import { InfiniteData, QueryClient } from "@tanstack/react-query";

import {
  getGetV2ListFavoriteLibraryAgentsQueryKey,
  getV2ListFavoriteLibraryAgentsResponse,
} from "@/app/api/__generated__/endpoints/library/library";
import { getGetV2ListLibraryAgentsResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { updateFavoriteInQueries } from "../helpers";

type FavoritesData = InfiniteData<
  getV2ListFavoriteLibraryAgentsResponse,
  number | undefined
>;

function makeAgent(overrides: Partial<LibraryAgent>): LibraryAgent {
  return { ...getGetV2ListLibraryAgentsResponseMock().agents[0], ...overrides };
}

function favoritesData(agents: LibraryAgent[]): FavoritesData {
  return {
    pageParams: [1],
    pages: [
      {
        status: 200,
        headers: new Headers(),
        data: {
          agents,
          pagination: {
            total_items: agents.length,
            total_pages: 1,
            current_page: 1,
            page_size: 10,
          },
        },
      },
    ],
  };
}

function agentIds(data: FavoritesData | undefined) {
  const page = data?.pages[0];
  return page?.status === 200 ? page.data.agents.map((a) => a.id) : [];
}

const unsearchedKey = getGetV2ListFavoriteLibraryAgentsQueryKey({
  page: 1,
  page_size: 10,
});
const searchedKey = getGetV2ListFavoriteLibraryAgentsQueryKey({
  page: 1,
  page_size: 10,
  search_term: "invoice",
});

function setup() {
  const queryClient = new QueryClient();
  const existing = makeAgent({ id: "invoice-bot", name: "Invoice Bot" });
  queryClient.setQueryData(unsearchedKey, favoritesData([existing]));
  queryClient.setQueryData(searchedKey, favoritesData([existing]));
  return { queryClient, existing };
}

describe("updateFavoriteInQueries", () => {
  test("adding a favorite leaves searched favorites lists to the server", () => {
    const { queryClient } = setup();
    const added = makeAgent({ id: "weather-bot", name: "Weather Bot" });

    updateFavoriteInQueries({
      queryClient,
      agentId: added.id,
      agent: added,
      newIsFavorite: true,
    });

    expect(
      agentIds(queryClient.getQueryData<FavoritesData>(unsearchedKey)),
    ).toEqual(["weather-bot", "invoice-bot"]);
    expect(
      agentIds(queryClient.getQueryData<FavoritesData>(searchedKey)),
    ).toEqual(["invoice-bot"]);
  });

  test("removing a favorite drops it from every favorites list", () => {
    const { queryClient, existing } = setup();

    updateFavoriteInQueries({
      queryClient,
      agentId: existing.id,
      agent: existing,
      newIsFavorite: false,
    });

    expect(
      agentIds(queryClient.getQueryData<FavoritesData>(unsearchedKey)),
    ).toEqual([]);
    expect(
      agentIds(queryClient.getQueryData<FavoritesData>(searchedKey)),
    ).toEqual([]);
  });
});
