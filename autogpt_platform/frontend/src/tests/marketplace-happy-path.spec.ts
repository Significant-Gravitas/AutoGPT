import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import {
  clickRunButton,
  LibraryPage,
  waitForAgentPageLoad,
} from "./pages/library.page";
import { MarketplacePage } from "./pages/marketplace.page";

test.use({ storageState: E2E_AUTH_STATES.marketplace });

test("marketplace happy path: user can browse Marketplace and open an agent detail page", async ({
  page,
}) => {
  test.setTimeout(90000);

  const marketplacePage = new MarketplacePage(page);
  await marketplacePage.openFeaturedAgent();

  await expect(page.getByTestId("agent-description")).toBeVisible();
});

test("marketplace happy path: user can add a Marketplace agent to Library and run it", async ({
  page,
}) => {
  test.setTimeout(120000);

  const marketplacePage = new MarketplacePage(page);
  await marketplacePage.openRunnableAgent();

  await page.getByTestId("agent-add-library-button").click();
  await expect(page.getByText("Redirecting to your library...")).toBeVisible();
  await expect(page).toHaveURL(/\/library\/agents\//);

  await waitForAgentPageLoad(page);
  await clickRunButton(page);

  const libraryPage = new LibraryPage(page);
  await libraryPage.waitForRunToComplete();

  const runStatus = await libraryPage.getRunStatus();
  expect(runStatus).toBe("completed");
  await libraryPage.assertRunProducedOutput();
});
