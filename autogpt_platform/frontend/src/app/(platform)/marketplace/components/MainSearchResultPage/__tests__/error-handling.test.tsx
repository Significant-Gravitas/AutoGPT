import { describe, expect, test } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainSearchResultPage } from "../MainSearchResultPage";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListStoreAgentsMockHandler422,
  getGetV2ListStoreCreatorsMockHandler422,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { create500Handler } from "@/tests/integrations/helpers/create-500-handler";

const defaultProps = {
  searchTerm: "test-search",
  sort: undefined as undefined,
};

describe("MainSearchResultPage - Error Handling", () => {
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
