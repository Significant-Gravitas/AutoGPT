import { expect, test } from "@playwright/test";
import { getTestUserWithLibraryAgents } from "./credentials";
import { LoginPage } from "./pages/login.page";
import { MarketplacePage } from "./pages/marketplace.page";
import { hasMinCount, hasUrl, isVisible, matchesUrl } from "./utils/assertion";

// Marketplace tests for store agent search functionality
test.describe("Marketplace – Basic Functionality", () => {
  test("User can access marketplace page when logged out", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const marketplaceTitle = await marketplacePage.getMarketplaceTitle(page);
    await isVisible(marketplaceTitle);

    console.log(
      "User can access marketplace page when logged out test passed ✅",
    );
  });

  test("User can access marketplace page when logged in", async ({ page }) => {
    const loginPage = new LoginPage(page);
    const marketplacePage = new MarketplacePage(page);

    await loginPage.goto();
    const richUser = getTestUserWithLibraryAgents();
    await loginPage.login(richUser.email, richUser.password);
    await hasUrl(page, "/marketplace");

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const marketplaceTitle = await marketplacePage.getMarketplaceTitle(page);
    await isVisible(marketplaceTitle);

    console.log(
      "User can access marketplace page when logged in test passed ✅",
    );
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

    console.log(
      "Featured agents, top agents, and featured creators are visible test passed ✅",
    );
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

    console.log(
      "Can navigate and interact with marketplace elements test passed ✅",
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

    await page.waitForTimeout(10000);

    const results = await marketplacePage.getSearchResultsCount(page);
    expect(results).toBeGreaterThan(0);

    console.log("Complete search flow works correctly test passed ✅");
  });

  // We need to add a test search with filters, but the current business logic for filters doesn't work as expected. We'll add it once we modify that.
});

test.describe("Marketplace – Edge Cases", () => {
  test("Search for non-existent item shows no results", async ({ page }) => {
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

    console.log("Search for non-existent item shows no results test passed ✅");
  });
});
