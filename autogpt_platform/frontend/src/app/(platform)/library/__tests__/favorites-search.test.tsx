import { describe, expect, test } from "vitest";
import userEvent from "@testing-library/user-event";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
  getGetV2ListFavoriteLibraryAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListLibraryFoldersMockHandler } from "@/app/api/__generated__/endpoints/folders/folders.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import LibraryPage from "../page";

const TOTAL_FAVORITES = 25;

function makeAgent(overrides: Partial<LibraryAgent> = {}): LibraryAgent {
  const base = getGetV2ListLibraryAgentsResponseMock().agents[0];
  return { ...base, ...overrides };
}

function setupHandlers() {
  const favoriteSearches: (string | null)[] = [];
  const favoritesPage = (page: number) =>
    Array.from({ length: 10 }, (_, i) => {
      const n = (page - 1) * 10 + i;
      return makeAgent({ id: `fav-${n}`, name: `Loaded Favorite ${n}` });
    }).slice(0, Math.max(0, TOTAL_FAVORITES - (page - 1) * 10));
  const unloadedFavorite = makeAgent({
    id: "fav-unloaded",
    name: "Invoice Sorter",
    is_favorite: true,
  });

  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      agents: [makeAgent({ id: "all-1", name: "Plain Agent" })],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler(({ request }) => {
      const params = new URL(request.url).searchParams;
      const searchTerm = params.get("search_term");
      const page = Number(params.get("page") ?? 1);
      favoriteSearches.push(searchTerm);
      if (searchTerm) {
        return {
          agents: [unloadedFavorite],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: 10,
          },
        };
      }
      return {
        agents: favoritesPage(page),
        pagination: {
          total_items: TOTAL_FAVORITES,
          total_pages: 3,
          current_page: page,
          page_size: 10,
        },
      };
    }),
    getGetV2ListLibraryFoldersMockHandler({
      folders: [],
      pagination: {
        total_items: 0,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
  );

  return { favoriteSearches };
}

function getFavoritesTab() {
  return screen
    .getAllByRole("tab")
    .find((t) => t.textContent?.match(/favorites/i));
}

describe("LibraryPage favorites search", () => {
  test("the Favorites badge shows the server total, not the loaded page", async () => {
    setupHandlers();

    render(<LibraryPage />);

    await waitFor(() =>
      expect(getFavoritesTab()?.textContent).toContain(
        `Favorites ${TOTAL_FAVORITES}`,
      ),
    );
  });

  test("searching on the Favorites tab finds a favorite that is not loaded yet", async () => {
    const { favoriteSearches } = setupHandlers();

    render(<LibraryPage />);

    await waitFor(() =>
      expect(getFavoritesTab()?.hasAttribute("data-disabled")).toBe(false),
    );
    await userEvent.click(getFavoritesTab()!);
    expect(await screen.findByText("Loaded Favorite 0")).toBeDefined();
    expect(screen.queryByText("Invoice Sorter")).toBeNull();

    await userEvent.type(
      screen.getAllByTestId("library-textbox")[0],
      "Invoice",
    );

    expect(await screen.findByText("Invoice Sorter")).toBeDefined();
    expect(favoriteSearches).toContain("Invoice");
    await waitFor(() =>
      expect(screen.queryByText("Loaded Favorite 0")).toBeNull(),
    );
    expect(getFavoritesTab()?.textContent).toContain("Favorites 1");
  });
});
