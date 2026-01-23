import { describe, expect, test, afterEach } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainSearchResultPage } from "../MainSearchResultPage";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListStoreAgentsMockHandler422,
  getGetV2ListStoreCreatorsMockHandler422,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { create500Handler } from "@/tests/integrations/helpers/create-500-handler";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const defaultProps = {
  searchTerm: "test-search",
  sort: undefined as undefined,
};

describe("MainSearchResultPage", () => {
  afterEach(() => {
    resetAuthState();
  });

  describe("rendering", () => {
    test("renders search results header with search term", async () => {
      render(<MainSearchResultPage {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Results for:")).toBeInTheDocument();
      });
      expect(screen.getByText("test-search")).toBeInTheDocument();
    });

    test("renders search bar", async () => {
      render(<MainSearchResultPage {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
      });
    });
  });

  describe("auth state", () => {
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

  describe("error handling", () => {
    test("displays error when agents API returns 422", async () => {
      server.use(getGetV2ListStoreAgentsMockHandler422());

      render(<MainSearchResultPage {...defaultProps} />);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load marketplace data", { exact: false }),
        ).toBeInTheDocument();
      });
    });

    test("displays error when creators API returns 422", async () => {
      server.use(getGetV2ListStoreCreatorsMockHandler422());

      render(<MainSearchResultPage {...defaultProps} />);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load marketplace data", { exact: false }),
        ).toBeInTheDocument();
      });
    });

    test("displays error when API returns 500", async () => {
      server.use(create500Handler("get", "*/api/store/agents*"));

      render(<MainSearchResultPage {...defaultProps} />);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load marketplace data", { exact: false }),
        ).toBeInTheDocument();
      });
    });

    test("retry button is visible on error", async () => {
      server.use(getGetV2ListStoreAgentsMockHandler422());

      render(<MainSearchResultPage {...defaultProps} />);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /try again/i }),
        ).toBeInTheDocument();
      });
    });
  });
});
