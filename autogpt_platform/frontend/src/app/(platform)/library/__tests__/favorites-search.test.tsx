import { describe, expect, test } from "vitest";
import userEvent from "@testing-library/user-event";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getPatchV2UpdateLibraryAgentMockHandler,
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
  const unfavorited = new Set<string>();
  const favoritesPage = (page: number) =>
    Array.from({ length: 10 }, (_, i) => {
      const n = (page - 1) * 10 + i;
      return makeAgent({
        id: `fav-${n}`,
        name: `Loaded Favorite ${n}`,
        is_favorite: true,
      });
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
        const matches = searchTerm === "Invoice" ? [unloadedFavorite] : [];
        return {
          agents: matches,
          pagination: {
            total_items: matches.length,
            total_pages: 1,
            current_page: 1,
            page_size: 10,
          },
        };
      }
      return {
        agents: favoritesPage(page).filter((a) => !unfavorited.has(a.id)),
        pagination: {
          total_items: TOTAL_FAVORITES - unfavorited.size,
          total_pages: 3,
          current_page: page,
          page_size: 10,
        },
      };
    }),
    getPatchV2UpdateLibraryAgentMockHandler(async ({ request, params }) => {
      const body = (await request.json()) as { is_favorite?: boolean };
      const id = String(params.libraryAgentId);
      if (body.is_favorite === false) unfavorited.add(id);
      return makeAgent({ id, is_favorite: body.is_favorite ?? true });
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

  test("unfavoriting brings the Favorites badge back in line with the server", async () => {
    setupHandlers();

    render(<LibraryPage />);

    await waitFor(() =>
      expect(getFavoritesTab()?.hasAttribute("data-disabled")).toBe(false),
    );
    await userEvent.click(getFavoritesTab()!);
    await screen.findByText("Loaded Favorite 0");

    const [firstHeart] = screen.getAllByRole("button", {
      name: "Remove from favorites",
    });
    await userEvent.click(firstHeart);

    await waitFor(() =>
      expect(getFavoritesTab()?.textContent).toContain(
        `Favorites ${TOTAL_FAVORITES - 1}`,
      ),
    );
  });
});
