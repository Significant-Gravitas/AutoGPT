import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainCreatorPage } from "../MainCreatorPage";
import { server } from "@/mocks/mock-server";
import {
  getGetV2GetCreatorDetailsMockHandler422,
  getGetV2ListStoreAgentsMockHandler422,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { create500Handler } from "@/tests/integrations/helpers/create-500-handler";

const defaultParams = {
  creator: "test-creator",
};

describe("MainCreatorPage - Error Handling", () => {
  test("displays error when creator details API returns 422", async () => {
    server.use(getGetV2GetCreatorDetailsMockHandler422());

    render(<MainCreatorPage params={defaultParams} />);

    expect(
      await screen.findByText("Failed to load creator data", { exact: false }),
    ).toBeInTheDocument();
  });

  test("displays error when creator agents API returns 422", async () => {
    server.use(getGetV2ListStoreAgentsMockHandler422());

    render(<MainCreatorPage params={defaultParams} />);

    expect(
      await screen.findByText("Failed to load creator data", { exact: false }),
    ).toBeInTheDocument();
  });

  test("displays error when API returns 500", async () => {
    server.use(create500Handler("get", "*/api/store/creator/test-creator"));

    render(<MainCreatorPage params={defaultParams} />);

    expect(
      await screen.findByText("Failed to load creator data", { exact: false }),
    ).toBeInTheDocument();
  });

  test("retry button is visible on error", async () => {
    server.use(getGetV2GetCreatorDetailsMockHandler422());

    render(<MainCreatorPage params={defaultParams} />);

    expect(
      await screen.findByRole("button", { name: /try again/i }),
    ).toBeInTheDocument();
  });
});
