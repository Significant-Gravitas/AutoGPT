import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";

describe("MainMarketplacePage - Empty State", () => {
  test("handles empty featured agents gracefully", async () => {
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

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Featured creators", { exact: false }),
    ).toBeInTheDocument();
  });

  test("handles empty creators gracefully", async () => {
    server.use(
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

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Featured agents", { exact: false }),
    ).toBeInTheDocument();
  });

  test("handles all empty data gracefully", async () => {
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

    render(<MainMarkeplacePage />);

    expect(await screen.findByPlaceholderText(/search/i)).toBeInTheDocument();
  });
});
