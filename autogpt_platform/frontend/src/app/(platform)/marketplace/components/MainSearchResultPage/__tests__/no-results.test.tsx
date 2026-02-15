import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainSearchResultPage } from "../MainSearchResultPage";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";

const defaultProps = {
  searchTerm: "nonexistent-search-term-xyz",
  sort: undefined,
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

    expect(await screen.findByText("Results for:")).toBeInTheDocument();
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

    expect(
      await screen.findByText("nonexistent-search-term-xyz"),
    ).toBeInTheDocument();
  });

  test("search bar is present with no results", async () => {
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

    expect(await screen.findByPlaceholderText(/search/i)).toBeInTheDocument();
  });
});
