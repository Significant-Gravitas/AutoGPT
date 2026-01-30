import { describe, expect, test, afterEach } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import LibraryPage from "../../page";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse, delay } from "msw";
import {
  mockAuthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

describe("LibraryPage - Loading State", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("shows loading spinner while agents are being fetched", async () => {
    mockAuthenticatedUser();

    // Override handlers to add delay to simulate loading
    server.use(
      http.get("*/api/library/agents*", async () => {
        await delay(500);
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
      http.get("*/api/library/agents/favorites*", async () => {
        await delay(500);
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

    const { container } = render(<LibraryPage />);

    // Check for loading spinner (LoadingSpinner component)
    const loadingElements = container.querySelectorAll(
      '[class*="animate-spin"]',
    );
    expect(loadingElements.length).toBeGreaterThan(0);
  });
});
