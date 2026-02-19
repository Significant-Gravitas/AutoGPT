import { describe, expect, test, afterEach } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainSearchResultPage } from "../MainSearchResultPage";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const defaultProps = {
  searchTerm: "test-search",
  sort: undefined,
};

describe("MainSearchResultPage - Auth State", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders page correctly when logged out", async () => {
    mockUnauthenticatedUser();
    render(<MainSearchResultPage {...defaultProps} />);

    expect(await screen.findByText("Results for:")).toBeInTheDocument();
  });

  test("renders page correctly when logged in", async () => {
    mockAuthenticatedUser();
    render(<MainSearchResultPage {...defaultProps} />);

    expect(await screen.findByText("Results for:")).toBeInTheDocument();
  });
});
