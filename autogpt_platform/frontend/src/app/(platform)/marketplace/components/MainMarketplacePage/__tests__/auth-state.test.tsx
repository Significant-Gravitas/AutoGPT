import { describe, expect, test, afterEach } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

describe("MainMarketplacePage - Auth State", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders page correctly when logged out", async () => {
    mockUnauthenticatedUser();
    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Featured agents", { exact: false }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Top Agents", { exact: false }),
    ).toBeInTheDocument();
  });

  test("renders page correctly when logged in", async () => {
    mockAuthenticatedUser();
    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Featured agents", { exact: false }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Top Agents", { exact: false }),
    ).toBeInTheDocument();
  });
});
