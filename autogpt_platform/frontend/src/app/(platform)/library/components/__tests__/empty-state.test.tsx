import { describe, expect, test, afterEach } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import LibraryPage from "../../page";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import {
  mockAuthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

describe("LibraryPage - Empty State", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("handles empty agents list gracefully", async () => {
    mockAuthenticatedUser();

    server.use(
      http.get("*/api/library/agents*", () => {
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

    render(<LibraryPage />);

    // Page should still render without crashing
    // Search bar should be visible even with no agents
    expect(
      await screen.findByPlaceholderText(/search agents/i),
    ).toBeInTheDocument();

    // Upload button should be visible
    expect(
      screen.getByRole("button", { name: /upload agent/i }),
    ).toBeInTheDocument();
  });

  test("handles empty favorites gracefully", async () => {
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

    render(<LibraryPage />);

    // Page should still render without crashing
    expect(
      await screen.findByPlaceholderText(/search agents/i),
    ).toBeInTheDocument();
  });
});
