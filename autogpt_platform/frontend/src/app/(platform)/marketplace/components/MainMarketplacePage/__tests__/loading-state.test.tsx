import { describe, expect, test } from "vitest";
import { render, waitFor } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse, delay } from "msw";

describe("MainMarketplacePage - Loading State", () => {
  test("shows loading skeleton while data is being fetched", async () => {
    server.use(
      http.get("*/api/store/agents*", async () => {
        await delay(500);
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
      http.get("*/api/store/creators*", async () => {
        await delay(500);
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

    const { container } = render(<MainMarkeplacePage />);

    await waitFor(() => {
      const loadingElements = container.querySelectorAll(
        '[class*="animate-pulse"]',
      );
      expect(loadingElements.length).toBeGreaterThan(0);
    });
  });
});
