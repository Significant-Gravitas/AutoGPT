import { describe, expect, test, afterEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainCreatorPage } from "../MainCreatorPage";
import { server } from "@/mocks/mock-server";
import {
  getGetV2GetCreatorDetailsMockHandler422,
  getGetV2ListStoreAgentsMockHandler422,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { create500Handler } from "@/tests/integrations/helpers/create-500-handler";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const defaultParams = {
  creator: "test-creator",
};

describe("MainCreatorPage", () => {
  afterEach(() => {
    resetAuthState();
  });

  describe("rendering", () => {
    test("renders creator info card", async () => {
      render(<MainCreatorPage params={defaultParams} />);
      await waitFor(() => {
        expect(screen.getByTestId("creator-description")).toBeInTheDocument();
      });
    });

    test("renders breadcrumbs with marketplace link", async () => {
      render(<MainCreatorPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByRole("link", { name: /marketplace/i }),
        ).toBeInTheDocument();
      });
    });

    test("renders about section", async () => {
      render(<MainCreatorPage params={defaultParams} />);

      await waitFor(() => {
        expect(screen.getByText("About")).toBeInTheDocument();
      });
    });

    test("renders agents by creator section", async () => {
      render(<MainCreatorPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByText(/Agents by/i, { exact: false }),
        ).toBeInTheDocument();
      });
    });
  });

  describe("auth state", () => {
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

  describe("error handling", () => {
    test("displays error when creator details API returns 422", async () => {
      server.use(getGetV2GetCreatorDetailsMockHandler422());

      render(<MainCreatorPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load creator data", { exact: false }),
        ).toBeInTheDocument();
      });
    });

    test("displays error when creator agents API returns 422", async () => {
      server.use(getGetV2ListStoreAgentsMockHandler422());

      render(<MainCreatorPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load creator data", { exact: false }),
        ).toBeInTheDocument();
      });
    });

    test("displays error when API returns 500", async () => {
      server.use(create500Handler("get", "*/api/store/creator/test-creator"));

      render(<MainCreatorPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load creator data", { exact: false }),
        ).toBeInTheDocument();
      });
    });

    test("retry button is visible on error", async () => {
      server.use(getGetV2GetCreatorDetailsMockHandler422());

      render(<MainCreatorPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /try again/i }),
        ).toBeInTheDocument();
      });
    });
  });
});
