import { expect, test } from "@playwright/test";
import { MarketplacePage } from "./pages/marketplace.page";
import { isVisible, matchesUrl } from "./utils/assertion";

test.describe("Marketplace – Navigation", () => {
  test("Can navigate and interact with marketplace elements", async ({
    page,
  }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto(page);

    const firstFeaturedAgent =
      await marketplacePage.getFirstFeaturedAgent(page);
    await firstFeaturedAgent.click();
    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
    await marketplacePage.goto(page);

    const firstTopAgent = await marketplacePage.getFirstTopAgent();
    await firstTopAgent.click();
    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
    await marketplacePage.goto(page);

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();
    await page.waitForURL("**/marketplace/creator/**");
    await matchesUrl(page, /\/marketplace\/creator\/.+/);

    console.log(
      "Can navigate and interact with marketplace elements test passed",
    );
  });

  test("Complete search flow works correctly", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto(page);

    await marketplacePage.searchAndNavigate("DummyInput", page);

    await marketplacePage.waitForSearchResults();

    await matchesUrl(page, /\/marketplace\/search\?searchTerm=/);

    const resultsHeading = page.getByText("Results for:");
    await isVisible(resultsHeading);

    const searchTerm = page.getByText("DummyInput").first();
    await isVisible(searchTerm);

    await page.waitForLoadState("networkidle").catch(() => {});

    await page
      .waitForFunction(
        () =>
          document.querySelectorAll('[data-testid="store-card"]').length > 0,
        { timeout: 15000 },
      )
      .catch(() => console.log("No search results appeared within timeout"));

    const results = await marketplacePage.getSearchResultsCount(page);
    expect(results).toBeGreaterThan(0);

    console.log("Complete search flow works correctly test passed");
  });
});
