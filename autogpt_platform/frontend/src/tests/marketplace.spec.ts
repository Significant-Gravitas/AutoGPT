import { test, expect } from "@playwright/test";
import { MarketplacePage } from "./pages/marketplace.page";
import { LoginPage } from "./pages/login.page";
import {
  isVisible,
  hasUrl,
  hasMinCount,
  matchesUrl,
} from "./utils/assertion";
import { TEST_CREDENTIALS } from "./credentials";

test.describe("Marketplace – Basic Functionality", () => {
  test("User can access marketplace page when logged out", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto();
    await hasUrl(page, "/marketplace");

    const marketplaceTitle = await marketplacePage.getMarketplaceTitle();
    await isVisible(marketplaceTitle);
  });

  test("User can access marketplace page when logged in", async ({ page }) => {
    const loginPage = new LoginPage(page);
    const marketplacePage = new MarketplacePage(page);

    await loginPage.goto();
    await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
    await hasUrl(page, "/marketplace");

    await marketplacePage.goto();
    await hasUrl(page, "/marketplace");

    const marketplaceTitle = await marketplacePage.getMarketplaceTitle();
    await isVisible(marketplaceTitle);
  });

  test("Featured agents, top agents, and featured creators are visible", async ({
    page,
  }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto();

    const featuredAgentsSection =
      await marketplacePage.getFeaturedAgentsSection();
    await isVisible(featuredAgentsSection);
    const featuredAgentCards = await marketplacePage.getFeaturedAgentCards();
    await hasMinCount(featuredAgentCards, 1);

    const topAgentsSection = await marketplacePage.getTopAgentsSection();
    await isVisible(topAgentsSection);
    const topAgentCards = await marketplacePage.getTopAgentCards();
    await hasMinCount(topAgentCards, 1);

    const featuredCreatorsSection =
      await marketplacePage.getFeaturedCreatorsSection();
    await isVisible(featuredCreatorsSection);
    const creatorProfiles = await marketplacePage.getCreatorProfiles();
    await hasMinCount(creatorProfiles, 1);
  });

  test("Can navigate and interact with marketplace elements", async ({
    page,
  }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto();

    const firstFeaturedAgent = await marketplacePage.getFirstFeaturedAgent();
    await firstFeaturedAgent.waitFor({ state: "visible" });
    await firstFeaturedAgent.click();
    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
    await marketplacePage.goto();

    const firstTopAgent = await marketplacePage.getFirstTopAgent();
    await firstTopAgent.click();
    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
    await marketplacePage.goto();

    const firstCreatorProfile = await marketplacePage.getFirstCreatorProfile();
    await firstCreatorProfile.click();
    await page.waitForURL("**/marketplace/creator/**");
    await matchesUrl(page, /\/marketplace\/creator\/.+/);
  });

  test("Complete search flow works correctly", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto();

    await marketplacePage.searchAndNavigate("DummyInput");

    await marketplacePage.waitForSearchResults();

    await matchesUrl(page, /\/marketplace\/search\?searchTerm=/);

    const resultsHeading = page.getByText("Results for:");
    await isVisible(resultsHeading);

    const searchTerm = page.getByText("DummyInput").first();
    await isVisible(searchTerm);

    const results = await marketplacePage.getSearchResultsCount();
    expect(results).toBeGreaterThan(1);
  });

  // We need to add a test search with filters, but the current business logic for filters doesn't work as expected. We'll add it once we modify that.
});

test.describe("Marketplace – Edge Cases", () => {
  test("Search for non-existent item shows no results", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);
    await marketplacePage.goto();

    await marketplacePage.searchAndNavigate("xyznonexistentitemxyz123");

    await marketplacePage.waitForSearchResults();

    await matchesUrl(page, /\/marketplace\/search\?searchTerm=/);

    const resultsHeading = page.getByText("Results for:");
    await isVisible(resultsHeading);

    const searchTerm = page.getByText("xyznonexistentitemxyz123");
    await isVisible(searchTerm);

    const results = await marketplacePage.getSearchResultsCount();
    expect(results).toBe(0);
  });
});
