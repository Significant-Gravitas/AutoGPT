import { describe, expect, test, afterEach } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { FavoritesSection } from "../FavoritesSection/FavoritesSection";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import {
  mockAuthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const mockFavoriteAgent = {
  id: "fav-agent-id",
  graph_id: "fav-graph-id",
  graph_version: 1,
  owner_user_id: "test-owner-id",
  image_url: null,
  creator_name: "Test Creator",
  creator_image_url: "https://example.com/avatar.png",
  status: "READY",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  name: "Favorite Agent Name",
  description: "Test favorite agent",
  input_schema: {},
  output_schema: {},
  credentials_input_schema: null,
  has_external_trigger: false,
  has_human_in_the_loop: false,
  has_sensitive_action: false,
  new_output: false,
  can_access_graph: true,
  is_latest_version: true,
  is_favorite: true,
};

describe("FavoritesSection", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders favorites section when there are favorites", async () => {
    mockAuthenticatedUser();

    server.use(
      http.get("*/api/library/agents/favorites*", () => {
        return HttpResponse.json({
          agents: [mockFavoriteAgent],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: 20,
          },
        });
      }),
    );

    render(<FavoritesSection searchTerm="" />);

    await waitFor(() => {
      expect(screen.getByText(/favorites/i)).toBeInTheDocument();
    });
  });

  test("renders favorite agent cards", async () => {
    mockAuthenticatedUser();

    server.use(
      http.get("*/api/library/agents/favorites*", () => {
        return HttpResponse.json({
          agents: [mockFavoriteAgent],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: 20,
          },
        });
      }),
    );

    render(<FavoritesSection searchTerm="" />);

    await waitFor(() => {
      expect(screen.getByText("Favorite Agent Name")).toBeInTheDocument();
    });
  });

  test("shows agent count", async () => {
    mockAuthenticatedUser();

    server.use(
      http.get("*/api/library/agents/favorites*", () => {
        return HttpResponse.json({
          agents: [mockFavoriteAgent],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: 20,
          },
        });
      }),
    );

    render(<FavoritesSection searchTerm="" />);

    await waitFor(() => {
      expect(screen.getByTestId("agents-count")).toBeInTheDocument();
    });
  });

  test("does not render when there are no favorites", async () => {
    mockAuthenticatedUser();

    server.use(
      http.get("*/api/library/agents/favorites*", () => {
        return HttpResponse.json({
          agents: [],
          pagination: {
            total_items: 0,
            total_pages: 0,
            current_page: 1,
            page_size: 20,
          },
        });
      }),
    );

    const { container } = render(<FavoritesSection searchTerm="" />);

    // Wait for loading to complete
    await waitFor(() => {
      // Component should return null when no favorites
      expect(container.textContent).toBe("");
    });
  });

  test("filters favorites based on search term", async () => {
    mockAuthenticatedUser();

    // Mock that returns different results based on search term
    server.use(
      http.get("*/api/library/agents/favorites*", ({ request }) => {
        const url = new URL(request.url);
        const searchTerm = url.searchParams.get("search_term");

        if (searchTerm === "nonexistent") {
          return HttpResponse.json({
            agents: [],
            pagination: {
              total_items: 0,
              total_pages: 0,
              current_page: 1,
              page_size: 20,
            },
          });
        }

        return HttpResponse.json({
          agents: [mockFavoriteAgent],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: 20,
          },
        });
      }),
    );

    const { rerender } = render(<FavoritesSection searchTerm="" />);

    await waitFor(() => {
      expect(screen.getByText("Favorite Agent Name")).toBeInTheDocument();
    });

    // Rerender with search term that yields no results
    rerender(<FavoritesSection searchTerm="nonexistent" />);

    await waitFor(() => {
      expect(screen.queryByText("Favorite Agent Name")).not.toBeInTheDocument();
    });
  });
});
