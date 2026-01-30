import { test } from "@playwright/test";
import { getTestUserWithLibraryAgents } from "./credentials";
import { LoginPage } from "./pages/login.page";
import { MarketplacePage } from "./pages/marketplace.page";
import { hasUrl, isVisible, matchesUrl } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";

test.describe("Marketplace Creator Page – Basic Functionality", () => {
  test("User can access creator's page when logged out", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();

    await page.waitForURL("**/marketplace/creator/**");
    await matchesUrl(page, /\/marketplace\/creator\/.+/);
  });

  test("User can access creator's page when logged in", async ({ page }) => {
    const loginPage = new LoginPage(page);
    const marketplacePage = new MarketplacePage(page);

    await loginPage.goto();
    const richUser = getTestUserWithLibraryAgents();
    await loginPage.login(richUser.email, richUser.password);
    await hasUrl(page, "/marketplace");

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();

    await page.waitForURL("**/marketplace/creator/**");
    await matchesUrl(page, /\/marketplace\/creator\/.+/);
  });

  test("Creator page details are visible", async ({ page }) => {
    const { getId } = getSelectors(page);
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();
    await page.waitForURL("**/marketplace/creator/**");

    const creatorTitle = getId("creator-title");
    await isVisible(creatorTitle);

    const creatorDescription = getId("creator-description");
    await isVisible(creatorDescription);
  });

  test("Agents in agent by sections navigation works", async ({ page }) => {
    const marketplacePage = new MarketplacePage(page);

    await marketplacePage.goto(page);
    await hasUrl(page, "/marketplace");

    const firstCreatorProfile =
      await marketplacePage.getFirstCreatorProfile(page);
    await firstCreatorProfile.click();
    await page.waitForURL("**/marketplace/creator/**");
    await page.waitForLoadState("networkidle").catch(() => {});

    const firstAgent = page
      .locator('[data-testid="store-card"]:visible')
      .first();
    await firstAgent.waitFor({ state: "visible", timeout: 30000 });

    await firstAgent.click();
    await page.waitForURL("**/marketplace/agent/**");
    await matchesUrl(page, /\/marketplace\/agent\/.+/);
  });
});
