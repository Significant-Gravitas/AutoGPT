import { describe, expect, test } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainSearchResultPage } from "../MainSearchResultPage";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";

const defaultProps = {
  searchTerm: "nonexistent-search-term-xyz",
  sort: undefined as undefined,
};

describe("MainSearchResultPage - No Results", () => {
  test("shows empty state when no agents match search", async () => {
    server.use(
      http.get("*/api/store/agents*", () => {
        return HttpResponse.json({
          agents: [],
          pagination: {
            total_items: 0,
            total_pages: 0,
            current_page: 1,
            page_size: 10,
          },
        });
      }),
      http.get("*/api/store/creators*", () => {
        return HttpResponse.json({
          creators: [],
          pagination: {
            total_items: 0,
            total_pages: 0,
            current_page: 1,
            page_size: 10,
          },
        });
      }),
    );

    render(<MainSearchResultPage {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Results for:")).toBeInTheDocument();
    });

    // Verify search term is displayed
    expect(screen.getByText("nonexistent-search-term-xyz")).toBeInTheDocument();
  });

  test("displays search term even with no results", async () => {
    server.use(
      http.get("*/api/store/agents*", () => {
        return HttpResponse.json({
          agents: [],
          pagination: {
            total_items: 0,
            total_pages: 0,
            current_page: 1,
            page_size: 10,
          },
        });
      }),
    );

    render(<MainSearchResultPage {...defaultProps} />);

    await waitFor(() => {
      expect(
        screen.getByText("nonexistent-search-term-xyz"),
      ).toBeInTheDocument();
    });
  });

  test("search bar remains functional with no results", async () => {
    server.use(
      http.get("*/api/store/agents*", () => {
        return HttpResponse.json({
          agents: [],
          pagination: {
            total_items: 0,
            total_pages: 0,
            current_page: 1,
            page_size: 10,
          },
        });
      }),
    );

    render(<MainSearchResultPage {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
    });
  });
});
