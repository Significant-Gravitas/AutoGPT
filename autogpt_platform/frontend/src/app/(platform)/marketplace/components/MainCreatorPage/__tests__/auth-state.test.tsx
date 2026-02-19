import { describe, expect, test, afterEach } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainCreatorPage } from "../MainCreatorPage";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const defaultParams = {
  creator: "test-creator",
};

describe("MainCreatorPage - Auth State", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders page correctly when logged out", async () => {
    mockUnauthenticatedUser();
    render(<MainCreatorPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByTestId("creator-description")).toBeInTheDocument();
    });
  });

  test("renders page correctly when logged in", async () => {
    mockAuthenticatedUser();
    render(<MainCreatorPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByTestId("creator-description")).toBeInTheDocument();
    });
  });
});
