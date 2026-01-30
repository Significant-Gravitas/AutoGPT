import { describe, expect, test, afterEach } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import LibraryPage from "../../page";
import {
  mockAuthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

describe("LibraryPage - Rendering", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders search bar", async () => {
    mockAuthenticatedUser();
    render(<LibraryPage />);

    expect(
      await screen.findByPlaceholderText(/search agents/i),
    ).toBeInTheDocument();
  });

  test("renders upload agent button", async () => {
    mockAuthenticatedUser();
    render(<LibraryPage />);

    expect(
      await screen.findByRole("button", { name: /upload agent/i }),
    ).toBeInTheDocument();
  });

  test("renders agent cards when data is loaded", async () => {
    mockAuthenticatedUser();
    render(<LibraryPage />);

    // Wait for agent cards to appear (from mock data)
    await waitFor(() => {
      const agentCards = screen.getAllByTestId("library-agent-card");
      expect(agentCards.length).toBeGreaterThan(0);
    });
  });

  test("agent cards display agent name", async () => {
    mockAuthenticatedUser();
    render(<LibraryPage />);

    // Wait for agent cards and check they have names
    await waitFor(() => {
      const agentNames = screen.getAllByTestId("library-agent-card-name");
      expect(agentNames.length).toBeGreaterThan(0);
    });
  });

  test("agent cards have see runs link", async () => {
    mockAuthenticatedUser();
    render(<LibraryPage />);

    await waitFor(() => {
      const seeRunsLinks = screen.getAllByTestId(
        "library-agent-card-see-runs-link",
      );
      expect(seeRunsLinks.length).toBeGreaterThan(0);
    });
  });
});
