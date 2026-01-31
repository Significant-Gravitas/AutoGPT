import { describe, expect, test, afterEach } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainSearchResultPage } from "../MainSearchResultPage";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const defaultProps = {
  searchTerm: "test-search",
  sort: undefined as undefined,
};

describe("MainSearchResultPage - Auth State", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders page correctly when logged out", async () => {
    mockUnauthenticatedUser();
    render(<MainSearchResultPage {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Results for:")).toBeInTheDocument();
    });
  });

  test("renders page correctly when logged in", async () => {
    mockAuthenticatedUser();
    render(<MainSearchResultPage {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Results for:")).toBeInTheDocument();
    });
  });
});
