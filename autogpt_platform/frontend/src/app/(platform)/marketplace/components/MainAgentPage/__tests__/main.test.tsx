import { describe, expect, test, afterEach } from "vitest";
import {
  render,
  screen,
  waitFor,
  act,
} from "@/tests/integrations/test-utils";
import { MainAgentPage } from "../MainAgentPage";
import { server } from "@/mocks/mock-server";
import { getGetV2GetSpecificAgentMockHandler422 } from "@/app/api/__generated__/endpoints/store/store.msw";
import { create500Handler } from "@/tests/integrations/helpers/create-500-handler";
import {
  mockAuthenticatedUser,
  mockUnauthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const defaultParams = {
  creator: "test-creator",
  slug: "test-agent",
};

describe("MainAgentPage", () => {
  afterEach(() => {
    resetAuthState();
  });

  describe("rendering", () => {
    test("renders agent info with title", async () => {
      render(<MainAgentPage params={defaultParams} />);
      await waitFor(() => {
        expect(screen.getByTestId("agent-title")).toBeInTheDocument();
      });
    });

    test("renders agent creator info", async () => {
      render(<MainAgentPage params={defaultParams} />);

      await waitFor(() => {
        expect(screen.getByTestId("agent-creator")).toBeInTheDocument();
      });
    });

    test("renders agent description", async () => {
      render(<MainAgentPage params={defaultParams} />);

      await waitFor(() => {
        expect(screen.getByTestId("agent-description")).toBeInTheDocument();
      });
    });

    test("renders breadcrumbs with marketplace link", async () => {
      render(<MainAgentPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByRole("link", { name: /marketplace/i }),
        ).toBeInTheDocument();
      });
    });

    test("renders download button", async () => {
      render(<MainAgentPage params={defaultParams} />);

      await waitFor(() => {
        expect(screen.getByTestId("agent-download-button")).toBeInTheDocument();
      });
    });

    test("renders similar agents section", async () => {
      render(<MainAgentPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByText("Similar agents", { exact: false }),
        ).toBeInTheDocument();
      });
    });
  });

  describe("auth state", () => {
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
      expect(
        screen.getByTestId("agent-add-library-button"),
      ).toBeInTheDocument();
    });
  });

  describe("error handling", () => {
    test("displays error when agent API returns 422", async () => {
      server.use(getGetV2GetSpecificAgentMockHandler422());

      render(<MainAgentPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load agent data", { exact: false }),
        ).toBeInTheDocument();
      });

      await act(async () => {});
    });

    test("displays error when API returns 500", async () => {
      server.use(
        create500Handler("get", "*/api/store/agents/test-creator/test-agent"),
      );

      render(<MainAgentPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load agent data", { exact: false }),
        ).toBeInTheDocument();
      });

      await act(async () => {});
    });

    test("retry button is visible on error", async () => {
      server.use(getGetV2GetSpecificAgentMockHandler422());

      render(<MainAgentPage params={defaultParams} />);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /try again/i }),
        ).toBeInTheDocument();
      });

      await act(async () => {});
    });
  });
});
