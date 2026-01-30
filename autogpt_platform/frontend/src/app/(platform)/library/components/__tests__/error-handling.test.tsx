import { describe, expect, test, afterEach } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import LibraryPage from "../../page";
import { server } from "@/mocks/mock-server";
import {
  mockAuthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";
import { create500Handler } from "@/tests/integrations/helpers/create-500-handler";
import {
  getGetV2ListLibraryAgentsMockHandler422,
  getGetV2ListFavoriteLibraryAgentsMockHandler422,
} from "@/app/api/__generated__/endpoints/library/library.msw";

describe("LibraryPage - Error Handling", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("handles API 422 error gracefully", async () => {
    mockAuthenticatedUser();

    server.use(getGetV2ListLibraryAgentsMockHandler422());

    render(<LibraryPage />);

    // Page should still render without crashing
    // Search bar should be visible even with error
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/search agents/i)).toBeInTheDocument();
    });
  });

  test("handles favorites API 422 error gracefully", async () => {
    mockAuthenticatedUser();

    server.use(getGetV2ListFavoriteLibraryAgentsMockHandler422());

    render(<LibraryPage />);

    // Page should still render without crashing
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/search agents/i)).toBeInTheDocument();
    });
  });

  test("handles API 500 error gracefully", async () => {
    mockAuthenticatedUser();

    server.use(create500Handler("get", "*/api/library/agents*"));

    render(<LibraryPage />);

    // Page should still render without crashing
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/search agents/i)).toBeInTheDocument();
    });
  });
});
