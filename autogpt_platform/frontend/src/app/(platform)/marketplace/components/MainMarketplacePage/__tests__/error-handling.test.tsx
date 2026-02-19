import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListStoreAgentsMockHandler422,
  getGetV2ListStoreCreatorsMockHandler422,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { create500Handler } from "@/tests/integrations/helpers/create-500-handler";

describe("MainMarketplacePage - Error Handling", () => {
  test("displays error when featured agents API returns 422", async () => {
    server.use(getGetV2ListStoreAgentsMockHandler422());

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Failed to load marketplace data", {
        exact: false,
      }),
    ).toBeInTheDocument();
  });

  test("displays error when creators API returns 422", async () => {
    server.use(getGetV2ListStoreCreatorsMockHandler422());

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Failed to load marketplace data", {
        exact: false,
      }),
    ).toBeInTheDocument();
  });

  test("displays error when API returns 500", async () => {
    server.use(create500Handler("get", "*/api/store/agents*"));

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Failed to load marketplace data", {
        exact: false,
      }),
    ).toBeInTheDocument();
  });

  test("retry button is visible on error", async () => {
    server.use(getGetV2ListStoreAgentsMockHandler422());

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByRole("button", { name: /try again/i }),
    ).toBeInTheDocument();
  });
});
