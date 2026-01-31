import { describe, expect, test } from "vitest";
import { render, screen, waitFor, act } from "@/tests/integrations/test-utils";
import { MainAgentPage } from "../MainAgentPage";
import { server } from "@/mocks/mock-server";
import { getGetV2GetSpecificAgentMockHandler422 } from "@/app/api/__generated__/endpoints/store/store.msw";
import { create500Handler } from "@/tests/integrations/helpers/create-500-handler";

const defaultParams = {
  creator: "test-creator",
  slug: "test-agent",
};

describe("MainAgentPage - Error Handling", () => {
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
