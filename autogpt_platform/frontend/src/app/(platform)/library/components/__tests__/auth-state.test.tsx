import { describe, expect, test, afterEach } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import LibraryPage from "../../page";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

describe("LibraryPage - Auth State", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders page correctly when logged in", async () => {
    mockAuthenticatedUser();
    render(<LibraryPage />);

    // Wait for upload button text to appear (indicates page is rendered)
    expect(
      await screen.findByText("Upload agent", { exact: false }),
    ).toBeInTheDocument();

    // Search bar should be visible
    expect(screen.getByTestId("search-bar")).toBeInTheDocument();
  });

  test("renders page correctly when logged out", async () => {
    mockUnauthenticatedUser();
    render(<LibraryPage />);

    // Wait for upload button text to appear (indicates page is rendered)
    expect(
      await screen.findByText("Upload agent", { exact: false }),
    ).toBeInTheDocument();

    // Search bar should still be visible
    expect(screen.getByTestId("search-bar")).toBeInTheDocument();
  });
});
