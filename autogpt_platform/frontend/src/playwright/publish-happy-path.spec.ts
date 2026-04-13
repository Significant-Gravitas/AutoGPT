import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";
import { LibraryPage } from "./pages/library.page";
import { MarketplacePage } from "./pages/marketplace.page";

test.use({ storageState: E2E_AUTH_STATES.parallelA });

test("publish happy path: user can submit, track, and delete an agent submission from the dashboard", async ({
  page,
}) => {
  test.setTimeout(180000);

  const buildPage = new BuildPage(page);
  const libraryPage = new LibraryPage(page);
  const marketplacePage = new MarketplacePage(page);

  const { agentName: publishableAgentName } =
    await buildPage.createAndSaveSimpleAgent("Publish Flow Agent");

  await page.goto("/library");
  await libraryPage.waitForAgentsToLoad();
  await libraryPage.searchAgents(publishableAgentName);
  await libraryPage.waitForAgentsToLoad();

  const createdAgent = page
    .getByTestId("library-agent-card")
    .filter({ hasText: publishableAgentName })
    .first();
  await expect(createdAgent).toBeVisible({ timeout: 15000 });

  const { agentTitle, agentSlug } =
    await marketplacePage.submitAgentForReview(publishableAgentName);

  await page.getByTestId("view-progress-button").click();
  await expect(page).toHaveURL(/\/profile\/dashboard/);
  await expect(page.getByText("Agent dashboard")).toBeVisible();

  const submissionRow =
    await marketplacePage.waitForDashboardSubmission(agentTitle);
  await expect(
    submissionRow.getByTestId("agent-status"),
    `submission "${agentTitle}" should appear in the dashboard review-pending state`,
  ).toContainText(/awaiting review/i);
  await submissionRow.getByTestId("agent-table-row-actions").click();
  await expect(page.getByRole("menuitem", { name: "Edit" })).toBeVisible();

  // Delete the submission via the actions menu. The dashboard does not show
  // a confirmation dialog — clicking Delete fires the API directly. We then
  // assert the row is gone, proving the backend actually removed it (not
  // just the menu item disappeared).
  await page.getByRole("menuitem", { name: "Delete" }).click();

  await expect(
    page.getByTestId("agent-table-row").filter({ hasText: agentTitle }),
    `submission row "${agentTitle}" must be removed from the dashboard after delete`,
  ).toHaveCount(0, { timeout: 15000 });

  // Validate the deleted submission is no longer discoverable in Marketplace.
  await page.goto("/marketplace");
  const searchInput = page
    .locator('[data-testid="store-search-input"]:visible')
    .first();
  await expect(searchInput).toBeVisible({ timeout: 15000 });
  await searchInput.fill(agentSlug);
  await searchInput.press("Enter");
  await expect(page).toHaveURL(/\/marketplace\/search/);

  await expect(
    page
      .locator(
        '[data-testid="store-card"], [data-testid="featured-store-card"]',
      )
      .filter({ hasText: agentTitle }),
    `deleted submission "${agentTitle}" should not appear in marketplace results`,
  ).toHaveCount(0, { timeout: 15000 });
});
