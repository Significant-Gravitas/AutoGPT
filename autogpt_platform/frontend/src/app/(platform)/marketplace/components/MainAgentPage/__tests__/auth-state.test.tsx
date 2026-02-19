import { describe, expect, test, afterEach } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainAgentPage } from "../MainAgentPage";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const defaultParams = {
  creator: "test-creator",
  slug: "test-agent",
};

describe("MainAgentPage - Auth State", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("shows add to library button when authenticated", async () => {
    mockAuthenticatedUser();
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(
        screen.getByTestId("agent-add-library-button"),
      ).toBeInTheDocument();
    });
  });

  test("hides add to library button when not authenticated", async () => {
    mockUnauthenticatedUser();
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByTestId("agent-title")).toBeInTheDocument();
    });
    expect(
      screen.queryByTestId("agent-add-library-button"),
    ).not.toBeInTheDocument();
  });

  test("renders page correctly when logged out", async () => {
    mockUnauthenticatedUser();
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByTestId("agent-title")).toBeInTheDocument();
    });
    expect(screen.getByTestId("agent-download-button")).toBeInTheDocument();
  });

  test("renders page correctly when logged in", async () => {
    mockAuthenticatedUser();
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByTestId("agent-title")).toBeInTheDocument();
    });
    expect(screen.getByTestId("agent-add-library-button")).toBeInTheDocument();
  });
});
