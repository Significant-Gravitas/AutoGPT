import { test, expect } from "@playwright/test";
import { MarketplacePage } from "./pages/marketplace.page";
import { LoginPage } from "./pages/login.page";
import { isVisible, hasUrl, hasMinCount, matchesUrl } from "./utils/assertion";
import { TEST_CREDENTIALS } from "./credentials";

test.describe("Marketplace – Basic Functionality", () => {
  test("User can access marketplace page when logged out", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const marketplaceTitle = await marketplacePage.getMarketplaceTitle(page);
    await isVisible(marketplaceTitle);
  });

  test("User can access marketplace page when logged in", async ({ page }) => {
    const loginPage = new LoginPage(page);
    const marketplacePage = new MarketplacePage(page);

    await loginPage.goto();
    await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
    await hasUrl(page, "/marketplace");

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const marketplaceTitle = await marketplacePage.getMarketplaceTitle(page);
    await isVisible(marketplaceTitle);
  });

  test("Featured agents, top agents, and featured creators are visible", async ({
    page,
  }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto(page);

    const featuredAgentsSection =
      await marketplacePage.getFeaturedAgentsSection(page);
    await isVisible(featuredAgentsSection);
    const featuredAgentCards =
      await marketplacePage.getFeaturedAgentCards(page);
    await hasMinCount(featuredAgentCards, 1);

    const topAgentsSection = await marketplacePage.getTopAgentsSection(page);
    await isVisible(topAgentsSection);
    const topAgentCards = await marketplacePage.getTopAgentCards(page);
    await hasMinCount(topAgentCards, 1);

    const featuredCreatorsSection =
      await marketplacePage.getFeaturedCreatorsSection(page);
    await isVisible(featuredCreatorsSection);
    const creatorProfiles = await marketplacePage.getCreatorProfiles(page);
    await hasMinCount(creatorProfiles, 1);
  });

  test("Can navigate and interact with marketplace elements", async ({
    page,
  }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto(page);

    const firstFeaturedAgent =
      await marketplacePage.getFirstFeaturedAgent(page);
    await firstFeaturedAgent.waitFor({ state: "visible" });
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
  });

  test.skip("Complete search flow works correctly", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto(page);

    await marketplacePage.searchAndNavigate("DummyInput", page);

    await marketplacePage.waitForSearchResults();

    await matchesUrl(page, /\/marketplace\/search\?searchTerm=/);

    const resultsHeading = page.getByText("Results for:");
    await isVisible(resultsHeading);

    const searchTerm = page.getByText("DummyInput").first();
    await isVisible(searchTerm);

    const results = await marketplacePage.getSearchResultsCount(page);
    expect(results).toBeGreaterThan(0);
  });

  // We need to add a test search with filters, but the current business logic for filters doesn't work as expected. We'll add it once we modify that.
});

test.describe("Marketplace – Edge Cases", () => {
  test.skip("Search for non-existent item shows no results", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto(page);

    await marketplacePage.searchAndNavigate("xyznonexistentitemxyz123", page);

    await marketplacePage.waitForSearchResults();

    await matchesUrl(page, /\/marketplace\/search\?searchTerm=/);

    const resultsHeading = page.getByText("Results for:");
    await isVisible(resultsHeading);

    const searchTerm = page.getByText("xyznonexistentitemxyz123");
    await isVisible(searchTerm);

    const results = await marketplacePage.getSearchResultsCount(page);
    expect(results).toBe(0);
  });
});
