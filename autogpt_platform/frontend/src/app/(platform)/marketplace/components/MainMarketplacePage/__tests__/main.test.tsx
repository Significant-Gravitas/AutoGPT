import { expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import { server } from "@/mocks/mock-server";
import { getDeleteV2DeleteStoreSubmissionMockHandler422 } from "@/app/api/__generated__/endpoints/store/store.msw";

// Only for CI testing purpose, will remove it in future PR
test("MainMarketplacePage", async () => {
  server.use(getDeleteV2DeleteStoreSubmissionMockHandler422());

  render(<MainMarkeplacePage />);
  expect(
    await screen.findByText("Featured agents", { exact: false }),
  ).toBeDefined();
});
